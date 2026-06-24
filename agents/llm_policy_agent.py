"""
agents/llm_policy_agent.py

LLM-encoded REINFORCE agent for cooperative Stag Hunt.

Architecture:
    text prompt -> Frozen Qwen3-4B (HuggingFace) -> mean-pooled hidden state -> LayerNorm -> PolicyHead MLP -> action probabilities (4,)

Only PolicyHead is trained. Qwen is always frozen.
Hidden dim is detected automatically from the model config at load time.

Changes from v1:
    - Mean pooling over all token positions instead of last-token only
      (more stable embedding across short templated prompts)
    - LayerNorm before the first Linear in PolicyHead
      (LLM hidden states have large magnitude variance that saturates Linear)
    - Entropy coefficient reduced from 0.01 -> 0.001
      (prevents entropy term from dominating and keeping policy uniform)
    - shaped_reward removed entirely
      (agents receive only raw env rewards; strategy is not injected)
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
    Converts a text prompt into a (hidden_dim,) float32 hidden state vector.

    Uses MEAN POOLING over all token positions in the last hidden layer.
    This is more stable than last-token for short, templated prompts where
    the last token is often punctuation or whitespace with low information.

    hidden_dim is read from model.config.hidden_size automatically,
    so this works with any Qwen variant (4B=2560, 7B=3584, etc.).

    Usage:
        encoder = LLMEncoder()
        h = encoder.encode("You are Agent A on a 5x5 grid ...")
        # h.shape == (2560,) for Qwen3-4B
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

        # Auto-detect hidden dim from model config — no hardcoding needed
        self.hidden_dim = self.model.config.hidden_size
        print(f"[LLMEncoder] Ready. hidden_dim={self.hidden_dim}")

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        """
        input:  str prompt (~200 tokens)
        output: Tensor shape (hidden_dim,) dtype float32 on CPU

        Mean-pools the last hidden layer over all token positions.
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
        # Last layer [-1], batch 0 -> (seq_len, hidden_dim)
        # Mean pool over seq_len -> (hidden_dim,)
        hidden = outputs.hidden_states[-1][0, -1, :]  # last token of last layer
        return hidden.float().cpu()


# ---------------------------------------------------------------------------
# 2. PolicyHead MLP  (the only trainable component)
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """
    Maps a frozen LLM hidden state to an action probability distribution.

    LayerNorm is applied first to normalise the LLM embedding before the
    first Linear layer. Without this, large variance in hidden state
    magnitudes saturates the linear weights and slows / prevents learning.

    input:  (batch, hidden_dim) float32
    output: (batch, 4)          float32 -- probabilities summing to 1
    """

    def __init__(self, hidden_dim: int, n_actions: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),   # normalise LLM embedding magnitude
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

    Both agents receive the SHARED TEAM REWARD (raw_r_a + raw_r_b) at each
    step. No reward shaping is applied — the agents must discover the
    cooperative strategy purely from the environment signal.

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
        lr: float    = 1e-4,    # reduced from 1e-3: LLM embeddings have unpredictable gradient scale
        gamma: float = 0.99,
    ):
        self.encoder  = encoder
        self.agent_id = agent_id
        self.gamma    = gamma

        # PolicyHead uses the actual hidden_dim from the loaded model
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
        probs = self.head(hidden.unsqueeze(0))      # (1, 4)

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
        loss = -sum( log_prob(a_t) * G_t ) - entropy_coeff * H(pi)

        G_t > 0: gradient increases prob of a_t (it was good)
        G_t < 0: gradient decreases prob of a_t (it was bad)

        Entropy bonus H(pi) prevents policy collapse but uses a small
        coefficient (0.001) so it does not dominate the loss and keep the
        policy artificially uniform.

        Normalization is skipped when std < 1e-6 (all returns identical)
        to avoid loss collapsing to zero and killing the gradient.

        Backprop flows through PolicyHead only. Qwen is untouched.
        Returns loss value for logging.
        """
        G, returns = 0.0, []
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Normalize only when there is meaningful variance.
        if len(returns_t) > 1 and returns_t.std().item() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs = torch.stack(self._log_probs)        # (T,)
        policy_loss = -(log_probs * returns_t).sum()

        # Entropy bonus: reduced coefficient 0.001 (was 0.01).
        # At 0.01, entropy dominated the loss when the policy was nearly uniform
        # (H ≈ log(4) ≈ 1.386), preventing the policy_loss from ever driving
        # meaningful weight updates. At 0.001 the bonus is a gentle regulariser.
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum()
        entropy_coeff = 0.001

        loss = policy_loss - entropy_coeff * entropy

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
