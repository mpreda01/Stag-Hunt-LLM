"""
agents/llm_policy_agent.py

LLM-encoded REINFORCE agent for cooperative Stag Hunt.

Architecture:
    text prompt -> Frozen Qwen3-4B (HuggingFace) -> hidden state (2048,) -> PolicyHead MLP -> action probabilities (4,)

Only PolicyHead is trained. Qwen is always frozen.
Prompts are built via obs_to_prompt() from agents/qwen4b.py, same as the frozen evaluation pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_DIM    = 4
INT_TO_ACTION = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
DEFAULT_MODEL = "Qwen/Qwen3-4B"


# ---------------------------------------------------------------------------
# 1. LLM Encoder  (frozen, never updated)
# ---------------------------------------------------------------------------

class LLMEncoder:
    """
    Wraps a frozen Qwen3-4B model loaded in 4-bit quantization.
    Converts a text prompt into a (2048,) float32 hidden state vector.

    This is different from agents/qwen4b.py (which uses Ollama for text generation).
    Here we use HuggingFace Transformers to access the internal hidden states,
    which Ollama does not expose.

    Usage:
        encoder = LLMEncoder()
        h = encoder.encode("You are Agent A on a 5x5 grid ...")  # (2048,)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        self.device = device

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        print(f"[LLMEncoder] Loading {model_name} in 4-bit ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            output_hidden_states=True,
            device_map=device,
        )
        self.model.eval()
        # Detect hidden dim from model config — works for any Qwen variant
        # (Qwen3-4B is 2560, not 2048 as previously assumed)
        self.hidden_dim = self.model.config.hidden_size
        print(f"[LLMEncoder] Ready. hidden_dim={self.hidden_dim}")

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        """
        input:  str prompt (~200 tokens)
        output: Tensor shape (hidden_dim,) dtype float32 on CPU

        Takes the last hidden layer's last token position as the state embedding.
        No gradients flow — Qwen weights never change.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
        # Last layer [-1], batch 0, last token [-1] -> (hidden_dim,)
        hidden = outputs.hidden_states[-1][0, -1, :]
        return hidden.float().cpu()


# ---------------------------------------------------------------------------
# 2. PolicyHead MLP  (the only trainable component)
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """
    Maps a frozen LLM hidden state to an action probability distribution.

    input:  (batch, hidden_dim) float32   -- hidden_dim passed from LLMEncoder
    output: (batch, 4)          float32   -- probabilities summing to 1
    """

    def __init__(self, hidden_dim: int, n_actions: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# 3. REINFORCE Agent
# ---------------------------------------------------------------------------

class REINFORCEAgent:
    """
    One agent = frozen LLMEncoder + trainable PolicyHead.

    Two instances are created in main.py (one per player).
    They share the same LLMEncoder but have independent PolicyHead weights
    and optimizers -- they learn separately from their own perspective.

    Episode flow:
        for each step:
            hidden = agent.encode(prompt)            # frozen Qwen
            action = agent.select_action(hidden)     # samples from PolicyHead
            agent.store_reward(reward)               # called after env.step()

        loss = agent.update()   # called once at END of episode
    """

    def __init__(
        self,
        encoder: LLMEncoder,
        agent_id: str,          # "A" or "B"
        lr: float    = 1e-3,
        gamma: float = 0.99,
    ):
        self.encoder  = encoder
        self.agent_id = agent_id
        self.gamma    = gamma

        self.head      = PolicyHead(hidden_dim=encoder.hidden_dim)
        self.optimizer = optim.Adam(self.head.parameters(), lr=lr)

        # Accumulated per episode, cleared after update()
        self._log_probs: list[torch.Tensor] = []
        self._rewards:   list[float]        = []

        # History for plotting / checkpointing
        self.loss_history:           list[float] = []
        self.episode_return_history: list[float] = []

    def encode(self, prompt: str) -> torch.Tensor:
        """Encode a text prompt -> (hidden_dim,) hidden state via frozen Qwen."""
        return self.encoder.encode(prompt)

    def select_action(self, hidden: torch.Tensor, greedy: bool = False) -> int:
        """
        Sample an action from the policy distribution.

        Training (greedy=False): stochastic sample, stores log_prob.
        Evaluation (greedy=True): argmax, no log_prob stored.
        """
        probs = self.head(hidden.unsqueeze(0))      # (1, 4) action probabilities

        if greedy:
            return int(probs.argmax(dim=1).item())

        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action))
        return int(action.item())

    def store_reward(self, reward: float):
        """Store the reward received at the current timestep."""
        self._rewards.append(reward)

    def update(self) -> float:
        """
        REINFORCE policy gradient update. Called once at the END of each episode.

        G_t = r_t + gamma*r_{t+1} + ... (discounted return from step t)
        loss = -sum( log_prob(a_t) * G_t )

        G_t > 0: gradient increases prob of a_t (it was good)
        G_t < 0: gradient decreases prob of a_t (it was bad)

        Backprop flows through PolicyHead only. Qwen is untouched.
        Returns loss value for logging.
        """
        G, returns = 0.0, []
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs = torch.stack(self._log_probs)
        loss      = -(log_probs * returns_t).sum()

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

    @staticmethod
    def shaped_reward(
        obs_before: list,
        obs_after:  list,
        env_reward: float,
        coeff:      float = 0.1,
    ) -> float:
        """
        Dense reward shaping: small bonus for moving toward the stag.
        obs layout: [ax, ay, bx, by, sx, sy, p1x, p1y, ...]
        Without shaping, most steps return 0 and REINFORCE has no gradient signal.
        coeff=0.1 keeps shaping small relative to true rewards (+5/-5).
        """
        dist_before = abs(float(obs_before[0]) - float(obs_before[4])) + \
                      abs(float(obs_before[1]) - float(obs_before[5]))
        dist_after  = abs(float(obs_after[0])  - float(obs_after[4]))  + \
                      abs(float(obs_after[1])  - float(obs_after[5]))
        return env_reward + coeff * (dist_before - dist_after)

    def save(self, path: str):
        torch.save({
            "head":                   self.head.state_dict(),
            "optimizer":              self.optimizer.state_dict(),
            "loss_history":           self.loss_history,
            "episode_return_history": self.episode_return_history,
        }, path)
        print(f"[Agent {self.agent_id}] Saved -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.head.load_state_dict(ckpt["head"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.loss_history           = ckpt.get("loss_history", [])
        self.episode_return_history = ckpt.get("episode_return_history", [])
        print(f"[Agent {self.agent_id}] Loaded <- {path}")
