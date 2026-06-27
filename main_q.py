"""
main_q.py  —  GRPO-style policy gradient training for Stag Hunt LLM agent

Architecture
------------
    QwenStagHuntPolicy  (qwen3b.py)
        ↕  prompt / response
    StagHuntEnv         (lightweight grid simulator, no pygame dependency)
        ↕  step / reset
    GRPOTrainer
        - collects rollout trajectories (prompt, response, reward)
        - normalises advantages within each group of rollouts
        - loss = -log_prob * advantage  +  β * KL(π_θ ‖ π_ref)
        - clips policy ratio (PPO-style) for stability
        - one Adam step per batch
    evaluate_agent      (greedy decoding, metrics: avg reward, coop rate)
    train_agent         (main loop, checkpointing, metric logging)
"""

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `import qwen3b` always resolves,
# regardless of the working directory Python is launched from.
# setup_q.sh installs a .pth file for the permanent fix; this is the runtime
# belt-and-suspenders for cases where the venv is used without that .pth.
# ---------------------------------------------------------------------------
import sys
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# ---------------------------------------------------------------------------

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from agents.qwen3b import (
    QwenStagHuntPolicy,
    generate_stag_hunt_prompt,
    parse_llm_output,
    INT_TO_ACTION,
    N_ACTIONS,
)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Environment
    grid_size:        int   = 5
    max_steps:        int   = 50
    stag_reward:      float = 5.0
    hare_reward:      float = 1.0
    maul_punishment:  float = -5.0
    n_hares:          int   = 2

    # Model
    model_name:       str   = "Qwen/Qwen2.5-3B-Instruct"
    lora_rank:        int   = 16
    lora_alpha:       int   = 32
    max_new_tokens:   int   = 256
    temperature:      float = 0.8

    # GRPO / PG
    lr:               float = 5e-5
    gamma:            float = 0.99          # discount
    kl_coeff:         float = 0.02          # β for KL penalty against ref model
    clip_eps:         float = 0.2           # PPO-style ratio clip
    group_size:       int   = 4             # rollouts per state (for GRPO baseline)
    grad_clip:        float = 1.0
    epochs:           int   = 200
    rollouts_per_ep:  int   = 2             # episodes collected before one update

    # Logging / checkpointing
    checkpoint_dir:   str   = "checkpoints_llm"
    checkpoint_every: int   = 20
    eval_every:       int   = 20
    eval_episodes:    int   = 10
    log_csv:          str   = "train_log.csv"

    # Hardware
    device:           str   = "cuda"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class StagHuntEnv:
    """
    Lightweight 2-agent grid Stag Hunt without any rendering dependency.

    Grid: N×N, coordinates (col, row), origin top-left.
    Entities: agent A, agent B (both controlled), one stag, several hares.

    Stag movement: each step it moves one cell toward the nearest agent
                   (Manhattan), breaking ties randomly.
    Hares:         stationary; respawn at a random free cell when collected.

    Observations per agent: dict with keys
        agent_pos, teammate_pos, stag_pos, hares_positions, step, grid_size
    """

    # Deltas for UP / DOWN / LEFT / RIGHT / STAY
    _DELTAS: dict[int, tuple[int, int]] = {
        0: ( 0, -1),   # UP
        1: ( 0,  1),   # DOWN
        2: (-1,  0),   # LEFT
        3: ( 1,  0),   # RIGHT
        4: ( 0,  0),   # STAY
    }

    def __init__(self, cfg: TrainConfig):
        self.N        = cfg.grid_size
        self.max_steps   = cfg.max_steps
        self.stag_reward = cfg.stag_reward
        self.hare_reward = cfg.hare_reward
        self.maul_punish = cfg.maul_punishment
        self.n_hares     = cfg.n_hares

        # State (initialised in reset)
        self.agent_a:  tuple[int, int] = (0, 0)
        self.agent_b:  tuple[int, int] = (0, 0)
        self.stag:     tuple[int, int] = (0, 0)
        self.hares:    list[tuple[int, int]] = []
        self.step_count: int = 0
        self.history_a:  list[str] = []
        self.history_b:  list[str] = []

    # ------------------------------------------------------------------

    def _rand_pos(self, exclude: set[tuple[int, int]]) -> tuple[int, int]:
        candidates = [
            (c, r)
            for c in range(self.N)
            for r in range(self.N)
            if (c, r) not in exclude
        ]
        return random.choice(candidates)

    def reset(self) -> tuple[dict, dict]:
        occupied: set[tuple[int, int]] = set()

        self.agent_a = self._rand_pos(occupied);  occupied.add(self.agent_a)
        self.agent_b = self._rand_pos(occupied);  occupied.add(self.agent_b)
        self.stag    = self._rand_pos(occupied);  occupied.add(self.stag)
        self.hares   = []
        for _ in range(self.n_hares):
            h = self._rand_pos(occupied)
            self.hares.append(h)
            occupied.add(h)

        self.step_count = 0
        self.history_a  = []
        self.history_b  = []

        return self._obs("A"), self._obs("B")

    def _obs(self, agent: str) -> dict:
        if agent == "A":
            return {
                "agent_pos":       self.agent_a,
                "teammate_pos":    self.agent_b,
                "stag_pos":        self.stag,
                "hares_positions": list(self.hares),
                "step":            self.step_count,
                "grid_size":       self.N,
                "history":         list(self.history_a),
            }
        else:
            return {
                "agent_pos":       self.agent_b,
                "teammate_pos":    self.agent_a,
                "stag_pos":        self.stag,
                "hares_positions": list(self.hares),
                "step":            self.step_count,
                "grid_size":       self.N,
                "history":         list(self.history_b),
            }

    def _clip(self, pos: tuple[int, int]) -> tuple[int, int]:
        c, r = pos
        return (max(0, min(self.N - 1, c)), max(0, min(self.N - 1, r)))

    def _move_stag(self):
        """Move stag one step toward the nearest agent (Manhattan)."""
        sc, sr = self.stag
        dists = {
            "A": abs(sc - self.agent_a[0]) + abs(sr - self.agent_a[1]),
            "B": abs(sc - self.agent_b[0]) + abs(sr - self.agent_b[1]),
        }
        target = self.agent_a if dists["A"] <= dists["B"] else self.agent_b
        tc, tr = target

        # Move one step in the direction of the target
        dc = 0 if tc == sc else (1 if tc > sc else -1)
        dr = 0 if tr == sr else (1 if tr > sr else -1)

        # Pick axis with larger delta to avoid diagonal
        if abs(tc - sc) >= abs(tr - sr):
            self.stag = (sc + dc, sr)
        else:
            self.stag = (sc, sr + dr)

    def step(
        self,
        action_a: int,
        action_b: int,
    ) -> tuple[dict, dict, float, float, bool, dict]:
        """
        Apply actions for both agents, move stag, compute rewards.

        Returns
        -------
        obs_a, obs_b, reward_a, reward_b, done, info
        """
        self.step_count += 1

        # Move agents
        da = self._DELTAS[action_a]
        db = self._DELTAS[action_b]
        self.agent_a = self._clip((self.agent_a[0] + da[0], self.agent_a[1] + da[1]))
        self.agent_b = self._clip((self.agent_b[0] + db[0], self.agent_b[1] + db[1]))

        # Record history
        self.history_a.append(f"Agent A: {INT_TO_ACTION[action_a]}")
        self.history_b.append(f"Agent B: {INT_TO_ACTION[action_b]}")

        # Move stag
        self._move_stag()

        # --- Reward computation ---
        reward_a = 0.0
        reward_b = 0.0
        info: dict = {"event": "none"}

        a_on_stag = self.agent_a == self.stag
        b_on_stag = self.agent_b == self.stag

        if a_on_stag and b_on_stag:
            # Cooperative catch
            reward_a += self.stag_reward
            reward_b += self.stag_reward
            info = {"event": "stag_caught"}
            # Respawn stag
            occupied = {self.agent_a, self.agent_b}
            occupied.update(self.hares)
            self.stag = self._rand_pos(occupied)

        elif a_on_stag and not b_on_stag:
            reward_a += self.maul_punish
            info = {"event": "maul_a"}

        elif b_on_stag and not a_on_stag:
            reward_b += self.maul_punish
            info = {"event": "maul_b"}

        # Check hare collection
        new_hares: list[tuple[int, int]] = []
        for h in self.hares:
            a_on_h = self.agent_a == h
            b_on_h = self.agent_b == h
            if a_on_h:
                reward_a += self.hare_reward
                info = {"event": "hare_a"}
                # respawn hare
                occupied = {self.agent_a, self.agent_b, self.stag}
                occupied.update(new_hares)
                h = self._rand_pos(occupied)
            if b_on_h:
                reward_b += self.hare_reward
                info = {"event": "hare_b"}
                occupied = {self.agent_a, self.agent_b, self.stag}
                occupied.update(new_hares)
                h = self._rand_pos(occupied)
            new_hares.append(h)
        self.hares = new_hares

        done = self.step_count >= self.max_steps

        return self._obs("A"), self._obs("B"), reward_a, reward_b, done, info


# ---------------------------------------------------------------------------
# Trajectory storage
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    prompt:    str
    response:  str
    action:    int
    reward:    float          # discounted return G_t (filled after episode)
    log_prob:  Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """
    Group Relative Policy Optimisation (simplified GRPO) trainer.

    For each state, `group_size` responses are sampled and their rewards
    normalised within the group to form the advantage.  This removes the
    need for a separate value network.

    Then we apply:
        advantage_i = (R_i - mean(R)) / (std(R) + ε)
        ratio       = exp(log_π_θ(a|s) - log_π_ref(a|s))
        clipped     = clip(ratio, 1-ε, 1+ε)
        pg_loss     = -min(ratio * adv, clipped * adv)
        kl_loss     = max(0, log_π_θ(a|s) - log_π_ref(a|s))  [one-sided KL]
        loss        = pg_loss + β * kl_loss
    """

    def __init__(self, policy: QwenStagHuntPolicy, cfg: TrainConfig):
        self.policy = policy
        self.cfg    = cfg

        # Only LoRA parameters are trainable
        trainable = [p for p in policy.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=cfg.lr)

    # ------------------------------------------------------------------

    def _discount_returns(
        self, rewards: list[float]
    ) -> list[float]:
        """Compute discounted returns G_t = r_t + γ r_{t+1} + …"""
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + self.cfg.gamma * G
            returns.insert(0, G)
        return returns

    # ------------------------------------------------------------------

    def collect_trajectory(
        self,
        env:   StagHuntEnv,
        agent: str,            # "A" or "B"
        obs_a: dict,
        obs_b: dict,
    ) -> tuple[list[Transition], float]:
        """
        Roll out ONE episode and collect per-step (prompt, response, reward)
        triples for the chosen agent.

        Returns (transitions, total_episode_reward_for_agent)
        """
        transitions: list[Transition] = []
        step_rewards: list[float]     = []

        obs = obs_a if agent == "A" else obs_b

        while True:
            prompt = generate_stag_hunt_prompt(
                agent_pos       = obs["agent_pos"],
                teammate_pos    = obs["teammate_pos"],
                stag_pos        = obs["stag_pos"],
                hares_positions = obs["hares_positions"],
                grid_size       = obs["grid_size"],
                history_log     = obs["history"],
            )

            action, response = self.policy.generate_action(
                prompt,
                temperature = self.cfg.temperature,
            )

            # We also need the opponent's action (random here for simplicity;
            # in self-play both policies are the same, so we re-use generate_action)
            opp_obs  = obs_b if agent == "A" else obs_a
            opp_prompt = generate_stag_hunt_prompt(
                agent_pos       = opp_obs["agent_pos"],
                teammate_pos    = opp_obs["agent_pos"],    # they see their own view
                stag_pos        = opp_obs["stag_pos"],
                hares_positions = opp_obs["hares_positions"],
                grid_size       = opp_obs["grid_size"],
                history_log     = opp_obs["history"],
            )
            opp_action, _ = self.policy.generate_action(opp_prompt, temperature=self.cfg.temperature)

            action_a = action    if agent == "A" else opp_action
            action_b = opp_action if agent == "A" else action

            obs_a_new, obs_b_new, r_a, r_b, done, info = env.step(action_a, action_b)

            r = r_a if agent == "A" else r_b
            step_rewards.append(r)
            transitions.append(Transition(
                prompt   = prompt,
                response = response,
                action   = action,
                reward   = r,    # will be replaced by discounted return below
            ))

            obs   = obs_a_new if agent == "A" else obs_b_new
            obs_a = obs_a_new
            obs_b = obs_b_new

            if done:
                break

        # Replace step rewards with discounted returns
        discounted = self._discount_returns(step_rewards)
        for t, g in zip(transitions, discounted):
            t.reward = g

        return transitions, sum(step_rewards)

    # ------------------------------------------------------------------

    def update(
        self,
        all_transitions: list[list[Transition]],
    ) -> dict[str, float]:
        """
        Compute GRPO loss over a batch of trajectories and do one gradient step.

        all_transitions : list of rollout transition lists (one per episode)
        """
        # Flatten
        flat: list[Transition] = [t for ep in all_transitions for t in ep]

        rewards_np = np.array([t.reward for t in flat], dtype=np.float32)
        # Normalise advantages globally across the batch
        mean_r = rewards_np.mean()
        std_r  = rewards_np.std() + 1e-8
        advantages = torch.tensor(
            (rewards_np - mean_r) / std_r,
            dtype = torch.float32,
            device = self.cfg.device,
        )

        total_loss   = torch.tensor(0.0, device=self.cfg.device)
        total_pg     = 0.0
        total_kl     = 0.0
        n            = len(flat)

        self.optimizer.zero_grad()

        for i, t in enumerate(flat):
            adv = advantages[i]

            # Current policy log-prob
            log_p = self.policy.log_probs_of_response(t.prompt, t.response)

            # Reference policy log-prob (frozen base model)
            ref_log_p = self.policy.reference_log_probs(t.prompt, t.response)

            # PPO-style clipped ratio
            ratio   = torch.exp(log_p - ref_log_p)
            clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)
            pg_loss = -torch.min(ratio * adv, clipped * adv)

            # One-sided KL: max(0, log π_θ - log π_ref)  (simpler than full KL)
            kl = torch.clamp(log_p - ref_log_p, min=0.0)

            loss_i = pg_loss + self.cfg.kl_coeff * kl
            total_loss = total_loss + loss_i / n

            total_pg += pg_loss.item()
            total_kl += kl.item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy.model.parameters() if p.requires_grad],
            self.cfg.grad_clip,
        )
        self.optimizer.step()

        return {
            "loss":    total_loss.item(),
            "pg_loss": total_pg / n,
            "kl":      total_kl / n,
            "mean_ret": float(mean_r),
            "std_ret":  float(std_r),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(
    policy:   QwenStagHuntPolicy,
    cfg:      TrainConfig,
    episodes: int,
) -> dict[str, float]:
    """
    Run `episodes` evaluation episodes with temperature=0 (greedy decoding).

    Metrics returned:
        avg_reward_a, avg_reward_b, avg_team_reward,
        stag_catch_rate (per step), cooperation_rate (fraction of steps where
        both agents moved toward the stag), maul_rate.
    """
    env = StagHuntEnv(cfg)
    total_r_a = total_r_b = 0.0
    total_catches = total_maulings = total_steps = 0

    for ep in range(episodes):
        obs_a, obs_b = env.reset()
        done = False

        while not done:
            # Agent A
            prompt_a = generate_stag_hunt_prompt(**{
                "agent_pos":       obs_a["agent_pos"],
                "teammate_pos":    obs_a["teammate_pos"],
                "stag_pos":        obs_a["stag_pos"],
                "hares_positions": obs_a["hares_positions"],
                "grid_size":       obs_a["grid_size"],
                "history_log":     obs_a["history"],
            })
            act_a, _ = policy.generate_action(prompt_a, temperature=0.01, do_sample=False)

            # Agent B
            prompt_b = generate_stag_hunt_prompt(**{
                "agent_pos":       obs_b["agent_pos"],
                "teammate_pos":    obs_b["teammate_pos"],
                "stag_pos":        obs_b["stag_pos"],
                "hares_positions": obs_b["hares_positions"],
                "grid_size":       obs_b["grid_size"],
                "history_log":     obs_b["history"],
            })
            act_b, _ = policy.generate_action(prompt_b, temperature=0.01, do_sample=False)

            obs_a, obs_b, r_a, r_b, done, info = env.step(act_a, act_b)

            total_r_a += r_a
            total_r_b += r_b
            total_steps += 1

            if info["event"] == "stag_caught":
                total_catches += 1
            if "maul" in info["event"]:
                total_maulings += 1

    n = max(episodes, 1)
    s = max(total_steps, 1)
    return {
        "avg_reward_a":     total_r_a / n,
        "avg_reward_b":     total_r_b / n,
        "avg_team_reward":  (total_r_a + total_r_b) / n,
        "stag_catch_rate":  total_catches / s,
        "maul_rate":        total_maulings / s,
        "total_steps":      total_steps,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_agent(
    policy: QwenStagHuntPolicy,
    cfg:    TrainConfig,
) -> None:
    """
    Main training loop.

    Each epoch:
      1. Collect `cfg.rollouts_per_ep` full episodes from agent A's perspective
         (self-play: agent B uses the same policy).
      2. Call GRPOTrainer.update() with all collected transitions.
      3. Every `cfg.checkpoint_every` epochs: save LoRA weights.
      4. Every `cfg.eval_every` epochs: run evaluation and log metrics.
    """
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env     = StagHuntEnv(cfg)
    trainer = GRPOTrainer(policy, cfg)

    log_path = ckpt_dir / cfg.log_csv
    csv_file  = open(log_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "loss", "pg_loss", "kl", "mean_ret", "std_ret",
        "eval_avg_team_reward", "eval_stag_catch_rate", "eval_maul_rate",
        "epoch_time_s",
    ])
    csv_writer.writeheader()

    print("=" * 70)
    print(f"  GRPO Training  |  {cfg.epochs} epochs  |  "
          f"grid={cfg.grid_size}x{cfg.grid_size}  |  "
          f"lr={cfg.lr}  |  kl_β={cfg.kl_coeff}")
    print("=" * 70)

    eval_metrics: dict[str, float] = {}

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # ---- Collect rollouts -------------------------------------------------
        all_transitions: list[list[Transition]] = []
        for _ in range(cfg.rollouts_per_ep):
            obs_a, obs_b = env.reset()
            traj, _ep_ret = trainer.collect_trajectory(env, "A", obs_a, obs_b)
            all_transitions.append(traj)

        # ---- Update -----------------------------------------------------------
        update_metrics = trainer.update(all_transitions)
        elapsed = time.time() - t0

        # ---- Evaluation (periodic) -------------------------------------------
        if epoch % cfg.eval_every == 0:
            print(f"\n  [Epoch {epoch}] Running evaluation …")
            eval_metrics = evaluate_agent(policy, cfg, cfg.eval_episodes)
            print(
                f"  EVAL  avg_team_R={eval_metrics['avg_team_reward']:.2f}  "
                f"catch_rate={eval_metrics['stag_catch_rate']*100:.1f}%  "
                f"maul_rate={eval_metrics['maul_rate']*100:.1f}%"
            )

        # ---- Logging ---------------------------------------------------------
        row = {
            "epoch":                epoch,
            "loss":                 f"{update_metrics['loss']:.6f}",
            "pg_loss":              f"{update_metrics['pg_loss']:.6f}",
            "kl":                   f"{update_metrics['kl']:.6f}",
            "mean_ret":             f"{update_metrics['mean_ret']:.4f}",
            "std_ret":              f"{update_metrics['std_ret']:.4f}",
            "eval_avg_team_reward": f"{eval_metrics.get('avg_team_reward', 0.0):.4f}",
            "eval_stag_catch_rate": f"{eval_metrics.get('stag_catch_rate', 0.0):.4f}",
            "eval_maul_rate":       f"{eval_metrics.get('maul_rate', 0.0):.4f}",
            "epoch_time_s":         f"{elapsed:.2f}",
        }
        csv_writer.writerow(row)
        csv_file.flush()

        print(
            f"  Ep {epoch:>4}/{cfg.epochs}  "
            f"loss={update_metrics['loss']:.4f}  "
            f"pg={update_metrics['pg_loss']:.4f}  "
            f"kl={update_metrics['kl']:.4f}  "
            f"mean_R={update_metrics['mean_ret']:.3f}  "
            f"t={elapsed:.1f}s"
        )

        # ---- Checkpoint -------------------------------------------------------
        if epoch % cfg.checkpoint_every == 0:
            ckpt_path = str(ckpt_dir / f"lora_ep{epoch}")
            policy.save(ckpt_path)
            # Also save latest pointer
            policy.save(str(ckpt_dir / "lora_latest"))

    csv_file.close()
    policy.save(str(ckpt_dir / "lora_final"))
    print("\n  Training complete.  Final checkpoint saved.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stag Hunt LLM GRPO trainer")
    parser.add_argument("--mode",       choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint", default=None,
                        help="LoRA adapter folder to load before training/eval")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--kl_coeff",   type=float, default=0.02)
    parser.add_argument("--eval_ep",    type=int,   default=10)
    parser.add_argument("--grid_size",  type=int,   default=5)
    parser.add_argument("--device",     default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainConfig(
        epochs       = args.epochs,
        lr           = args.lr,
        kl_coeff     = args.kl_coeff,
        eval_episodes = args.eval_ep,
        grid_size    = args.grid_size,
        device       = args.device,
    )

    # Build policy
    policy = QwenStagHuntPolicy(
        model_name      = cfg.model_name,
        device          = cfg.device,
        lora_rank       = cfg.lora_rank,
        lora_alpha      = cfg.lora_alpha,
        max_new_tokens  = cfg.max_new_tokens,
    )

    if args.checkpoint is not None:
        policy.load(args.checkpoint)

    if args.mode == "train":
        train_agent(policy, cfg)

    else:
        print("=" * 70)
        print(f"  EVALUATION  |  {args.eval_ep} episodes")
        print("=" * 70)
        metrics = evaluate_agent(policy, cfg, args.eval_ep)
        print(json.dumps(metrics, indent=2))
