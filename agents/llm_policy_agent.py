"""
agents/llm_policy_agent.py  —  Option B: LLM Token-Logit REINFORCE

Architecture
------------
    text prompt
        -> Frozen Qwen3-4B (4-bit)
        -> next-token logits over vocab  (shape: vocab_size,)
        -> slice 4 action-token logits   [LEFT, DOWN, RIGHT, UP]
        -> TemperatureAdapter            (1 trainable scalar per agent)
        -> softmax -> action distribution

What is trained
---------------
    Only TemperatureAdapter.temperature  (1 scalar, ~0 overhead).
    Qwen weights are never touched.

Why this works where the hidden-state approach didn't
------------------------------------------------------
    The hidden-state approach produced identical embeddings for all states
    (L2 dist = 0.0) because mean-pooling or last-token over a ~600-token
    prompt dominated by the fixed preamble destroys state information.

    Here we bypass hidden states entirely. Instead we ask the LLM to
    predict the NEXT TOKEN after "ACTION:" and read the logits for the
    four action-word tokens directly from the output layer. These logits
    ARE state-dependent: the LLM's language model head assigns different
    probabilities to "RIGHT" vs "LEFT" depending on the coordinates in
    the prompt. The LLM's world knowledge is therefore fully exploited.

    REINFORCE then learns a single temperature scalar that scales the
    confidence of those logits to match the cooperative reward signal.
    - temperature > 1  ->  softer distribution (more exploration)
    - temperature < 1  ->  sharper distribution (more exploitation)
    - temperature = 1  ->  raw LLM distribution (baseline)

    This is minimal RL fine-tuning: the LLM reasons, RL calibrates.
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_WORDS  = ["LEFT", "DOWN", "RIGHT", "UP"]   # order matches env encoding
INT_TO_ACTION = {i: a for i, a in enumerate(ACTION_WORDS)}
ACTION_TO_INT = {a: i for i, a in enumerate(ACTION_WORDS)}
DEFAULT_MODEL = "Qwen/Qwen3-4B"


# ---------------------------------------------------------------------------
# 1.  LLM  (frozen, never updated)
# ---------------------------------------------------------------------------

class LLMPolicy:
    """
    Frozen Qwen3-4B loaded in 4-bit NF4 quantisation.

    Core method: get_action_logits(prompt) -> Tensor shape (4,)
        Runs one forward pass, reads the next-token logits at the position
        right after the final "ACTION:" token, then slices out the four
        action-word token ids.

    The returned logits are RAW (not normalised). Pass them to
    TemperatureAdapter to get a probability distribution.

    Token-id lookup
    ---------------
    We look up each action word once at init time. Qwen3 tokenises
    "LEFT", "DOWN", "RIGHT", "UP" as single tokens, so each maps to
    exactly one token id. We assert this at load time so mismatches
    surface immediately rather than silently producing wrong logits.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        self.device = device

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        print(f"[LLMPolicy] Loading {model_name} in 4-bit ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
        )
        self.model.eval()
        print(f"[LLMPolicy] Model loaded.")

        # Resolve action token ids — each word must be a SINGLE token
        self.action_token_ids = []
        for word in ACTION_WORDS:
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            assert len(ids) == 1, (
                f"Action word '{word}' tokenises to {len(ids)} tokens {ids}. "
                f"Qwen3 must tokenise it as a single token for logit slicing to work."
            )
            self.action_token_ids.append(ids[0])

        print(f"[LLMPolicy] Action token ids: "
              f"{ {w: i for w, i in zip(ACTION_WORDS, self.action_token_ids)} }")

    @torch.no_grad()
    def get_action_logits(self, prompt: str) -> torch.Tensor:
        """
        Returns raw logits for the 4 action tokens, shape (4,) on CPU float32.

        The prompt must end with 'ACTION:' (or similar) so that the model's
        next-token prediction naturally targets action words.

        Implementation note:
            outputs.logits has shape (1, seq_len, vocab_size).
            We take position [-1] (last input token) which gives the
            distribution over the NEXT token — i.e., what comes after ACTION:.
            We then index into vocab_size with our 4 action token ids.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.model(**inputs)

        # (1, seq_len, vocab_size) -> last position -> (vocab_size,)
        last_logits = outputs.logits[0, -1, :]

        # Slice the 4 action-word logits -> (4,)
        action_logits = torch.stack([
            last_logits[tid] for tid in self.action_token_ids
        ])

        return action_logits.float().cpu()


# ---------------------------------------------------------------------------
# 2.  TemperatureAdapter  (the ONLY trainable component)
# ---------------------------------------------------------------------------

class TemperatureAdapter(nn.Module):
    """
    Scales the LLM's action logits by a single learned temperature.

    forward(logits) -> probabilities, shape (4,)

    The temperature is initialised to 1.0 so the agent starts with the
    raw LLM distribution and RL moves it from there.

    Gradient flows ONLY through this scalar — Qwen is untouched.

    Why one scalar and not a full MLP?
        We proved (via the embedding diagnostic) that the LLM hidden states
        carry zero state information. A full MLP on top of those states
        cannot learn. The token logits ARE state-dependent (the LLM assigns
        different next-token probabilities for different coordinate values),
        so the minimal trainable component is enough: just calibrate the
        confidence of the LLM's own distribution.
    """

    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        # Use log-space so temperature is always positive after exp()
        self.log_temperature = nn.Parameter(
            torch.tensor([init_temperature]).log()
        )

    @property
    def temperature(self) -> float:
        return self.log_temperature.exp().item()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (4,) -> probs: (4,)"""
        temp = self.log_temperature.exp()          # always positive
        return torch.softmax(logits / temp, dim=0)


# ---------------------------------------------------------------------------
# 3.  REINFORCE Agent
# ---------------------------------------------------------------------------

class REINFORCEAgent:
    """
    One agent = frozen LLMPolicy + trainable TemperatureAdapter.

    Episode flow:
        for each step:
            logits = agent.get_action_logits(prompt)   # frozen Qwen
            action = agent.select_action(logits)        # samples via adapter
            agent.store_reward(reward)                  # after env.step()

        agent.update()   # called once at END of episode

    Two instances share the same LLMPolicy but have independent
    TemperatureAdapters — each agent calibrates its own confidence level.
    """

    def __init__(
        self,
        llm: LLMPolicy,
        agent_id: str,           # "A" or "B"
        lr: float    = 1e-2,     # higher lr is fine: only 1 param to update
        gamma: float = 0.99,
    ):
        self.llm      = llm
        self.agent_id = agent_id
        self.gamma    = gamma

        self.adapter   = TemperatureAdapter(init_temperature=1.0)
        self.optimizer = optim.Adam(self.adapter.parameters(), lr=lr)

        # Accumulated per episode, cleared after update()
        self._log_probs: list[torch.Tensor] = []
        self._rewards:   list[float]        = []

        # History for logging / checkpointing
        self.loss_history:           list[float] = []
        self.episode_return_history: list[float] = []

    def get_action_logits(self, prompt: str) -> torch.Tensor:
        """Get raw action logits from frozen LLM. Shape (4,)."""
        return self.llm.get_action_logits(prompt)

    def select_action(
        self,
        logits: torch.Tensor,
        greedy: bool = False,
    ) -> int:
        """
        Training (greedy=False): sample from adapter distribution, store log_prob.
        Evaluation (greedy=True): argmax, no log_prob stored.

        logits: (4,) raw action logits from LLMPolicy.get_action_logits()
        """
        probs = self.adapter(logits)   # (4,) — goes through temperature scaling

        if greedy:
            return int(probs.argmax().item())

        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action))
        return int(action.item())

    def store_reward(self, reward: float):
        self._rewards.append(reward)

    def update(self) -> float:
        """
        REINFORCE update at end of episode.

        loss = -sum( log_prob(a_t) * G_t ) - entropy_coeff * H(pi)

        Only TemperatureAdapter.log_temperature is updated.
        Returns loss value for logging.
        """
        # Compute discounted returns
        G, returns = 0.0, []
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Normalise returns when there is meaningful variance
        if len(returns_t) > 1 and returns_t.std().item() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs   = torch.stack(self._log_probs)         # (T,)
        policy_loss = -(log_probs * returns_t).sum()

        # Entropy bonus — small coefficient keeps distribution from collapsing
        probs   = log_probs.exp()
        entropy = -(probs * log_probs).sum()

        loss = policy_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        self.episode_return_history.append(sum(self._rewards))

        self._log_probs.clear()
        self._rewards.clear()

        return loss_val

    def save(self, path: str):
        torch.save({
            "adapter":                self.adapter.state_dict(),
            "optimizer":              self.optimizer.state_dict(),
            "loss_history":           self.loss_history,
            "episode_return_history": self.episode_return_history,
        }, path)
        print(f"[Agent {self.agent_id}] Saved -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.adapter.load_state_dict(ckpt["adapter"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.loss_history           = ckpt.get("loss_history", [])
        self.episode_return_history = ckpt.get("episode_return_history", [])
        print(f"[Agent {self.agent_id}] Loaded <- {path}")
