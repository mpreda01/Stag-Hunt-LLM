"""
agents/llm_policy_agent.py  —  Minimal-Prompt LLM Encoder + MLP Policy

Architecture
------------
    raw obs (10 floats)
        -> minimal text prompt  (~20 tokens, ~43% state signal)
        -> Frozen Qwen3-4B
        -> mean pool last hidden layer  -> (hidden_dim,)
        -> detach  (no gradient into Qwen)
        -> LayerNorm + MLP PolicyHead   (trainable)
        -> action probabilities (4,)

Why this fixes the embedding collapse
--------------------------------------
    Previous prompts had ~600 tokens of fixed preamble and ~10 state
    tokens — the state was 1.6% of the mean-pooled embedding and was
    completely drowned out.

    The minimal prompt has NO preamble at all:
        "agent:0,0 teammate:4,0 stag:2,2 plants:1,3,3,1 action:"
    This is ~20 tokens of which ~15 are state-specific coordinates.
    The state is now ~43-75% of the embedding, so different observations
    produce measurably different hidden states.

    We verified the old approach gave L2 dist = 0.0000 between ALL
    state pairs.  The new approach should give L2 dist >> 0.

Why mean pool + detach + MLP (not last-token, not LoRA)
--------------------------------------------------------
    - Mean pool: with ~20 tokens, mean pooling is meaningful (no long
      fixed prefix to drown out the state).
    - Detach: keeps Qwen frozen — no backward pass through 4B params,
      so this runs on RTX 2080 (11 GB) not just L40 (48 GB).
    - LayerNorm before MLP: LLM hidden states have large magnitude
      variance (~85 norm) that saturates linear layers without it.
    - MLP PolicyHead: now has genuinely different inputs per state,
      so gradients are non-zero and learning is possible.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_WORDS  = ["LEFT", "DOWN", "RIGHT", "UP"]
ACTION_TO_INT = {w: i for i, w in enumerate(ACTION_WORDS)}
INT_TO_ACTION = {i: w for i, w in enumerate(ACTION_WORDS)}
DEFAULT_MODEL = "Qwen/Qwen3-4B"


# ---------------------------------------------------------------------------
# Prompt builder  (minimal — state dominates)
# ---------------------------------------------------------------------------

def build_minimal_prompt(obs: np.ndarray, agent: str) -> str:
    """
    Convert a flat (10,) coordinate observation to a minimal text prompt.

    Format:
        "agent:ax,ay teammate:bx,by stag:sx,sy plants:p1x,p1y,p2x,p2y action:"

    Token count: ~20 tokens
    State signal: ~43-75% of total tokens (vs 1.6% in the old two-shot prompt)

    The word "action:" at the end acts as a trigger token — the model's
    representation at this position naturally attends to what came before,
    i.e., the coordinate values.
    """
    obs  = np.array(obs, dtype=float)
    ax, ay = int(obs[0]), int(obs[1])
    bx, by = int(obs[2]), int(obs[3])
    sx, sy = int(obs[4]), int(obs[5])

    # Remaining obs entries are plant coordinates
    plants = [int(obs[i]) for i in range(6, len(obs))]
    plants_str = ",".join(str(v) for v in plants)

    agent_label = agent.lower()   # "a" or "b"

    return (
        f"agent_{agent_label}:{ax},{ay} "
        f"teammate:{bx},{by} "
        f"stag:{sx},{sy} "
        f"plants:{plants_str} "
        f"respond with only one word: LEFT RIGHT UP or DOWN. action:"
    )


# ---------------------------------------------------------------------------
# 1.  Frozen LLM Encoder
# ---------------------------------------------------------------------------

class LLMEncoder:
    """
    Frozen Qwen3-4B in 4-bit NF4.

    encode(prompt) -> Tensor (hidden_dim,) on CPU, detached.

    Uses MEAN POOLING over all token positions of the last hidden layer.
    With the minimal prompt (~20 tokens), mean pooling gives a stable
    embedding that actually varies with the state coordinates.

    No gradients flow through this class — Qwen weights never change.
    Runs on RTX 2080 (11 GB) since there is no backward pass.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        self.device = device

        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = torch.float16,
            bnb_4bit_use_double_quant = True,
        )

        print(f"[LLMEncoder] Loading {model_name} in 4-bit NF4 ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb_config,
            output_hidden_states = True,
            device_map          = device,
        )
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size
        print(f"[LLMEncoder] Ready. hidden_dim={self.hidden_dim}")

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        """
        Returns mean-pooled last hidden layer as a (hidden_dim,) CPU tensor.
        Detached — no gradient flows into Qwen.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 128,    # minimal prompt fits easily in 128
        ).to(self.device)

        outputs = self.model(**inputs, output_hidden_states=True)

        # (1, seq_len, hidden_dim) -> mean over seq_len -> (hidden_dim,)
        hidden = outputs.hidden_states[-1][0]   # (seq_len, hidden_dim)
        hidden = hidden.mean(dim=0)             # (hidden_dim,)

        return hidden.float().cpu().detach()


# ---------------------------------------------------------------------------
# 2.  PolicyHead MLP  (only trainable component)
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """
    Maps a detached LLM embedding to an action probability distribution.

    LayerNorm first: LLM hidden states have norm ~85 and large per-dim
    variance. Without normalisation the first Linear layer saturates
    immediately and gradients vanish.

    input:  (batch, hidden_dim)
    output: (batch, 4)  — softmax probabilities
    """

    def __init__(self, hidden_dim: int, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# 3.  REINFORCE Agent
# ---------------------------------------------------------------------------

class REINFORCEAgent:
    """
    One agent = frozen LLMEncoder + trainable PolicyHead.

    Both agents share the same LLMEncoder but have independent
    PolicyHead weights and optimizers.

    Episode flow:
        hidden = agent.encode(obs, agent_id)   # frozen Qwen, minimal prompt
        action = agent.select_action(hidden)    # MLP samples action
        agent.store_reward(team_reward)         # after env.step()
        ...
        agent.update()                          # REINFORCE at end of episode

    Reward: shared team reward (raw_r_a + raw_r_b), no shaping.
    """

    def __init__(
        self,
        encoder:  LLMEncoder,
        agent_id: str,           # "A" or "B"
        lr:       float = 1e-3,
        gamma:    float = 0.99,
    ):
        self.encoder  = encoder
        self.agent_id = agent_id
        self.gamma    = gamma

        self.head      = PolicyHead(hidden_dim=encoder.hidden_dim)
        self.optimizer = optim.Adam(self.head.parameters(), lr=lr)

        self._log_probs: list[torch.Tensor] = []
        self._rewards:   list[float]        = []

        self.loss_history:           list[float] = []
        self.episode_return_history: list[float] = []

    def encode(self, obs: np.ndarray) -> torch.Tensor:
        """
        Build minimal prompt from raw obs and encode via frozen Qwen.
        Returns (hidden_dim,) tensor, detached, on CPU.
        """
        prompt = build_minimal_prompt(obs, agent=self.agent_id)
        return self.encoder.encode(prompt)

    def select_action(
        self,
        hidden:  torch.Tensor,
        greedy:  bool = False,
    ) -> int:
        probs = self.head(hidden.unsqueeze(0))   # (1, 4)

        if greedy:
            return int(probs.argmax(dim=1).item())

        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action))
        return int(action.item())

    def store_reward(self, reward: float):
        self._rewards.append(reward)

    def update(self) -> float:
        """
        REINFORCE update at end of episode.

        G_t = r_t + gamma * G_{t+1}
        loss = -sum(log_prob(a_t) * G_t) - entropy_coeff * H(pi)

        Returns loss value for logging.
        """
        G, returns = 0.0, []
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32)

        if len(returns_t) > 1 and returns_t.std().item() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs   = torch.stack(self._log_probs)
        policy_loss = -(log_probs * returns_t).sum()

        probs   = log_probs.exp()
        entropy = -(probs * log_probs).sum()

        loss = policy_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=5.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        self.episode_return_history.append(sum(self._rewards))

        self._log_probs.clear()
        self._rewards.clear()

        return loss_val

    def save(self, path: str):
        torch.save({
            "head":                   self.head.state_dict(),
            "optimizer":              self.optimizer.state_dict(),
            "loss_history":           self.loss_history,
            "episode_return_history": self.episode_return_history,
            "hidden_dim":             self.encoder.hidden_dim,
        }, path)
        print(f"[Agent {self.agent_id}] Saved -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.head.load_state_dict(ckpt["head"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.loss_history           = ckpt.get("loss_history", [])
        self.episode_return_history = ckpt.get("episode_return_history", [])
        print(f"[Agent {self.agent_id}] Loaded <- {path}")
