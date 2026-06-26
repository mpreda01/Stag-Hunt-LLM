"""
visual_policy.py  —  CNN-based DQN for Stag Hunt (image observations)

Architecture
------------
    RGB image (H, W, 3)
        -> normalize /255
        -> Conv layers (3->32->64)
        -> AdaptiveAvgPool to (64, 4, 4)
        -> Flatten -> Linear(1024, 256) -> ReLU -> Linear(256, 4)
        -> Q-values for 4 actions

Shared replay buffer design
----------------------------
    Both agents A and B push their transitions into the SAME ReplayBuffer
    inside DQNAgent. This means one CNN learns a single cooperative policy
    that works regardless of which "role" the agent plays. Both agents use
    identical obs (their own RGB frame) and receive the same team reward,
    so the shared policy learns to cooperate by maximizing joint return.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# 1. CNN Policy
# ---------------------------------------------------------------------------

class CNNPolicy(nn.Module):
    """
    Maps a single agent's RGB image observation to Q-values for 4 actions.

    Input:  (batch, H, W, 3)  uint8 or float32  — pixels in [0, 255]
    Output: (batch, 4)  float32 Q-values
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> (batch, 64, 4, 4)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                  # -> (batch, 64*4*4 = 1024)
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, H, W, 3)  float32 with values in [0, 255]
        Returns: (batch, 4) Q-values
        """
        # Normalize pixels to [0, 1]
        x = x / 255.0
        # Permute to (batch, C, H, W) for Conv2d
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 2. Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-size circular buffer storing (obs, action, reward, next_obs, done).

    obs / next_obs: float32 numpy arrays of shape (H, W, 3).
    """

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ):
        self.buffer.append((
            obs.astype(np.float32),
            int(action),
            float(reward),
            next_obs.astype(np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        """Return a random batch as stacked numpy arrays."""
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.stack(obs),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_obs),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 3. DQN Agent  (shared policy for both agents A and B)
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Single DQNAgent whose shared policy is trained from the combined
    experience of both agents A and B.

    Rationale: in a symmetric cooperative game like Hunt, both agents
    face structurally identical sub-problems (navigate to the stag). A
    single CNN trained on transitions from both agents learns a role-
    agnostic "converge on the stag" policy, doubling the effective
    sample rate at no extra memory cost.

    Online network: updated every step via Bellman MSE loss.
    Target network: hard-copied from online every target_update_freq steps.
    """

    def __init__(
        self,
        lr:                 float = 1e-4,
        gamma:              float = 0.99,
        epsilon_start:      float = 1.0,
        epsilon_end:        float = 0.05,
        epsilon_decay:      float = 0.995,
        target_update_freq: int   = 10,
        batch_size:         int   = 64,
        buffer_size:        int   = 10_000,
        device:             str   = "cpu",
    ):
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size         = batch_size
        self.device             = device

        # Online network (trained via backprop)
        self.online = CNNPolicy().to(device)
        # Target network (periodically synced, never backprop'd through)
        self.target = CNNPolicy().to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(capacity=buffer_size)

        # Training metadata
        self._update_count:   int         = 0
        self.loss_history:    list[float] = []
        self.return_history:  list[float] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """
        Epsilon-greedy during training (greedy=False),
        pure greedy during evaluation (greedy=True).

        obs: (H, W, 3) uint8 or float32 numpy array.
        """
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, 3)

        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online(obs_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------

    def store(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ):
        """Push a transition into the shared replay buffer."""
        self.buffer.push(obs, action, reward, next_obs, done)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self) -> float | None:
        """
        Sample a mini-batch and perform one gradient step on the online net.

        Target: y = r + gamma * max_a Q_target(s', a) * (1 - done)
        Loss:   MSE(Q_online(s, a), y)

        Returns loss value, or None if buffer is not yet full enough.
        """
        if len(self.buffer) < self.batch_size:
            return None

        obs_b, act_b, rew_b, next_obs_b, done_b = self.buffer.sample(self.batch_size)

        # Move everything to device
        obs_t      = torch.from_numpy(obs_b).to(self.device)          # (B, H, W, 3)
        act_t      = torch.from_numpy(act_b).long().to(self.device)    # (B,)
        rew_t      = torch.from_numpy(rew_b).to(self.device)           # (B,)
        next_obs_t = torch.from_numpy(next_obs_b).to(self.device)      # (B, H, W, 3)
        done_t     = torch.from_numpy(done_b).to(self.device)          # (B,)

        # Online Q-values for taken actions
        q_all    = self.online(obs_t)                                  # (B, 4)
        q_taken  = q_all.gather(1, act_t.unsqueeze(1)).squeeze(1)      # (B,)

        # Target Q-values (no gradient)
        with torch.no_grad():
            q_next  = self.target(next_obs_t).max(dim=1).values        # (B,)
            q_target = rew_t + self.gamma * q_next * (1.0 - done_t)   # (B,)

        loss = nn.functional.mse_loss(q_taken, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Decay epsilon multiplicatively after each update
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self._update_count += 1
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def update_target(self):
        """Hard copy online weights to target network."""
        self.target.load_state_dict(self.online.state_dict())

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "online_state_dict":    self.online.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon":              self.epsilon,
            "update_count":         self._update_count,
            "loss_history":         self.loss_history,
            "return_history":       self.return_history,
        }, path)
        print(f"[DQNAgent] Saved -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online_state_dict"])
        self.target.load_state_dict(ckpt["online_state_dict"])  # sync target too
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon         = ckpt.get("epsilon",        self.epsilon_end)
        self._update_count   = ckpt.get("update_count",   0)
        self.loss_history    = ckpt.get("loss_history",   [])
        self.return_history  = ckpt.get("return_history", [])
        self.target.eval()
        print(f"[DQNAgent] Loaded <- {path}  (epsilon={self.epsilon:.4f})")
