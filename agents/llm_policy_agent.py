"""
agents/llm_policy_agent.py

LLM-encoded REINFORCE agent for cooperative Stag Hunt.

Architecture:
    text prompt -> Frozen Qwen3-4B (HuggingFace) -> hidden state -> PolicyHead MLP -> action probabilities (4,)

Only PolicyHead is trained. Qwen is always frozen.
Hidden dim is detected automatically from the model config at load time.
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

    input:  (batch, hidden_dim) float32
    output: (batch, 4)          float32 -- probabilities summing to 1
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

        Entropy bonus H(pi) prevents policy collapse: it penalizes the policy
        for becoming too deterministic, forcing continued exploration.

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
        # When std ~ 0 (all returns identical), skip normalization —
        # otherwise loss collapses to 0 and weights never update.
        if len(returns_t) > 1 and returns_t.std().item() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs = torch.stack(self._log_probs)        # (T,)
        policy_loss = -(log_probs * returns_t).sum()

        # Entropy bonus: H(pi) = -sum(p * log(p))
        # Recompute probabilities from the stored log_probs for entropy.
        # coeff=0.01 keeps entropy small but prevents full collapse.
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum()
        entropy_coeff = 0.01

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

    @staticmethod
    def shaped_reward(
        obs_before: list,
        obs_after:  list,
        env_reward: float,
        coeff:      float = 0.3,
    ) -> float:
        """
        Dense reward shaping: small bonus for moving toward the stag.
        obs layout: [ax, ay, bx, by, sx, sy, p1x, p1y, ...]
        Without shaping, most steps return 0 and REINFORCE has no gradient signal.
        coeff=0.3 keeps shaping moderate relative to true rewards (+5/-5).
        """
        # Distance to stag
        dist_stag_before = abs(float(obs_before[0]) - float(obs_before[4])) + \
                        abs(float(obs_before[1]) - float(obs_before[5]))
        dist_stag_after  = abs(float(obs_after[0])  - float(obs_after[4]))  + \
                        abs(float(obs_after[1])  - float(obs_after[5]))
        stag_shaping = coeff * (dist_stag_before - dist_stag_after)

        # Distance to teammate — reward converging with partner
        dist_team_before = abs(float(obs_before[0]) - float(obs_before[2])) + \
                        abs(float(obs_before[1]) - float(obs_before[3]))
        dist_team_after  = abs(float(obs_after[0])  - float(obs_after[2]))  + \
                        abs(float(obs_after[1])  - float(obs_after[3]))
        team_shaping = coeff * (dist_team_before - dist_team_after)

        return env_reward + stag_shaping + team_shaping
    
    
   

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
