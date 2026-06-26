"""
main.py  —  DQN (CNN policy) training and evaluation on Stag Hunt

Architecture:
    RGB image obs (H, W, 3)
        -> CNNPolicy online/target networks
        -> shared ReplayBuffer (transitions from both agents A and B)
        -> team reward: r_team = r_A + r_B
        -> Bellman MSE loss on online net, hard target sync every N steps

Shared replay buffer rationale:
    Both agents see equivalent RGB frames (their own first-person view)
    and both receive the same team reward. Pooling their transitions into
    one buffer doubles the effective sample rate and forces the single
    CNN to learn a role-agnostic cooperative policy without any extra
    architectural complexity.

Key fix vs previous version:
    The environment is now created ONCE per train/eval run and reused
    across all episodes via env.reset(). The old design called make_env()
    and env.close() inside run_episode() on every episode, which caused
    HuntEnv to crash on re-initialisation (pygame/display state collision)
    after the first episode. set -e in run.sh then silently killed the job
    with only a tqdm "0/1000" line visible in the log.

Usage:
    python main.py --mode train
    python main.py --mode eval --checkpoint checkpoints/dqn_latest.pt
"""

import argparse
import traceback
import time
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

import torch
import numpy as np

from utils.const import ENV_FACTORIES
from utils.utils import (
    RolloutFrame,
    get_pixel_frame,
    save_rollout_video,
    save_rollout_csv,
)
from agents.visual_policy import DQNAgent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    # Environment
    "env_name":           "hunt",
    "grid_size":          (5, 5),
    "max_timesteps":      200,
    "stag_reward":        5,
    "forage_reward":      1,
    "mauling_punishment": -5,

    # Training
    "total_episodes": 1000,

    # DQN hyperparameters
    "lr":                 1e-4,
    "gamma":              0.99,
    "epsilon_start":      1.0,
    "epsilon_end":        0.05,
    "epsilon_decay":      0.995,
    "target_update_freq": 10,
    "batch_size":         64,
    "buffer_size":        10_000,

    # Checkpointing
    "checkpoint_dir":   "checkpoints",
    "checkpoint_every": 100,

    # Evaluation
    "eval_episodes": 20,
    "save_video":    True,
}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(load_renderer: bool = False):
    """
    Instantiate a Hunt environment with image observations.

    load_renderer=True is only needed when collecting pixel frames for
    video saving and costs extra overhead, so it defaults to False.

    Important: call this ONCE per training/eval run and reuse the returned
    env by calling env.reset() between episodes. Creating and closing a
    new env every episode causes a pygame display-state crash in HuntEnv
    on the second call to __init__.
    """
    return ENV_FACTORIES[CONFIG["env_name"]](
        obs_type           = "image",
        load_renderer      = load_renderer,
        enable_multiagent  = True,
        grid_size          = CONFIG["grid_size"],
        max_timesteps      = CONFIG["max_timesteps"],
        stag_reward        = CONFIG["stag_reward"],
        forage_reward      = CONFIG["forage_reward"],
        mauling_punishment = CONFIG["mauling_punishment"],
    )


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    agent:           DQNAgent,
    env,
    episode_idx:     int,
    training:        bool = True,
    collect_frames:  bool = False,
) -> list[RolloutFrame]:
    """
    Run one full episode with the DQN agent controlling both agents A and B.

    The env is passed in from outside and only reset() is called here —
    it is never closed. This avoids the pygame re-initialisation crash that
    occurred when make_env() / env.close() were called inside this function.

    collect_frames controls whether pixel frames are captured for video
    export. It requires load_renderer=True on the env and is expensive,
    so it is only enabled when explicitly requested (e.g. eval ep 1).

    Training flow per step:
      1. agent.select_action(obs_a)  — epsilon-greedy for A
      2. agent.select_action(obs_b)  — epsilon-greedy for B
      3. env.step([a_A, a_B])
      4. team_reward = r_A + r_B     — shared cooperative signal
      5. agent.store(obs_a, a_A, team_reward, next_obs_a, done)
      6. agent.store(obs_b, a_B, team_reward, next_obs_b, done)
         Both transitions go into the SAME replay buffer so the CNN
         trains on twice as many samples per wall-clock episode.
      7. agent.update()              — one gradient step on online net
      8. agent.update_target()       — hard sync every target_update_freq
                                       steps (skipped at step 0 to avoid
                                       a spurious sync before any learning)

    Evaluation flow: greedy actions, no store/update calls.
    """
    obs, info = env.reset()

    obs_a = np.array(obs[0], dtype=np.float32)
    obs_b = np.array(obs[1], dtype=np.float32)

    frames: list[RolloutFrame] = []
    frames.append(RolloutFrame(
        step        = 0,
        obs         = obs,
        actions     = None,
        rewards     = None,
        pixel_frame = get_pixel_frame(env, multiagent=True) if collect_frames else None,
        info        = info,
    ))

    mode_str   = "Train" if training else "Eval"
    step_range = range(1, CONFIG["max_timesteps"] + 1)
    step_iter  = (
        tqdm(step_range, desc=f"  {mode_str} ep {episode_idx}",
             unit="step", leave=False, dynamic_ncols=True)
        if TQDM_AVAILABLE else step_range
    )

    last_loss = 0.0

    for step in step_iter:
        step_t0 = time.time()

        action_a = agent.select_action(obs_a, greedy=not training)
        action_b = agent.select_action(obs_b, greedy=not training)

        next_obs, rewards, terminated, truncated, info = env.step([action_a, action_b])
        raw_r_a, raw_r_b = float(rewards[0]), float(rewards[1])

        next_obs_a = np.array(next_obs[0], dtype=np.float32)
        next_obs_b = np.array(next_obs[1], dtype=np.float32)

        done = bool(terminated or truncated)

        if training:
            team_reward = raw_r_a + raw_r_b

            agent.store(obs_a, action_a, team_reward, next_obs_a, done)
            agent.store(obs_b, action_b, team_reward, next_obs_b, done)

            loss = agent.update()
            if loss is not None:
                last_loss = loss

            # Hard-sync target network every target_update_freq update steps.
            # Guard _update_count > 0 so we don't sync before the first
            # gradient step (when _update_count starts at 0, 0 % N == 0).
            if agent._update_count > 0 and agent._update_count % CONFIG["target_update_freq"] == 0:
                agent.update_target()

        frames.append(RolloutFrame(
            step        = step,
            obs         = next_obs,
            actions     = [action_a, action_b],
            rewards     = (raw_r_a, raw_r_b),
            pixel_frame = get_pixel_frame(env, multiagent=True) if collect_frames else None,
            info        = info,
        ))

        if TQDM_AVAILABLE:
            step_iter.set_postfix({
                "r_A":  f"{raw_r_a:+.1f}",
                "r_B":  f"{raw_r_b:+.1f}",
                "team": f"{raw_r_a + raw_r_b:+.1f}",
                "eps":  f"{agent.epsilon:.3f}",
                "loss": f"{last_loss:.4f}",
                "t":    f"{time.time() - step_t0:.2f}s",
            })

        obs_a = next_obs_a
        obs_b = next_obs_b
        if done:
            break

    return frames


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(frames: list[RolloutFrame]) -> dict:
    total_r_a  = 0.0
    total_r_b  = 0.0
    n_catches  = 0
    n_maulings = 0

    stag_r = CONFIG["stag_reward"]
    maul_p = CONFIG["mauling_punishment"]

    for f in frames:
        if f.rewards is None:
            continue
        r_a, r_b = float(f.rewards[0]), float(f.rewards[1])
        total_r_a += r_a
        total_r_b += r_b
        if r_a == stag_r and r_b == stag_r:
            n_catches += 1
        if r_a == maul_p or r_b == maul_p:
            n_maulings += 1

    return {
        "total_reward_a":    total_r_a,
        "total_reward_b":    total_r_b,
        "total_team_reward": total_r_a + total_r_b,
        "n_catches":         n_catches,
        "n_maulings":        n_maulings,
        "steps":             len(frames) - 1,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print("=" * 65)
    print("  TRAINING: Shared DQN (CNN policy) on Stag Hunt")
    print(f"  lr={CONFIG['lr']} | gamma={CONFIG['gamma']}")
    print(f"  eps: {CONFIG['epsilon_start']} -> {CONFIG['epsilon_end']} "
          f"(decay={CONFIG['epsilon_decay']})")
    print("=" * 65)

    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    agent = DQNAgent(
        lr                 = CONFIG["lr"],
        gamma              = CONFIG["gamma"],
        epsilon_start      = CONFIG["epsilon_start"],
        epsilon_end        = CONFIG["epsilon_end"],
        epsilon_decay      = CONFIG["epsilon_decay"],
        target_update_freq = CONFIG["target_update_freq"],
        batch_size         = CONFIG["batch_size"],
        buffer_size        = CONFIG["buffer_size"],
        device             = device,
    )

    ckpt_latest   = ckpt_dir / "dqn_latest.pt"
    start_episode = 1
    if ckpt_latest.exists():
        agent.load(str(ckpt_latest))
        start_episode = len(agent.return_history) + 1
        print(f"Resuming from episode {start_episode}\n")

    total_eps = CONFIG["total_episodes"]

    # Create the environment once and reuse it for the entire training run.
    # Calling env.reset() between episodes is sufficient and avoids the
    # pygame/display re-initialisation crash that happens when you call
    # make_env() + env.close() inside the loop every episode.
    print("Creating environment...")
    env = make_env(load_renderer=False)
    print("Environment ready.\n")

    ep_range = range(start_episode, total_eps + 1)
    ep_iter  = (
        tqdm(ep_range, desc="Training episodes", unit="ep", dynamic_ncols=True)
        if TQDM_AVAILABLE else ep_range
    )

    try:
        for episode in ep_iter:
            ep_t0 = time.time()

            try:
                frames = run_episode(
                    agent       = agent,
                    env         = env,
                    episode_idx = episode,
                    training    = True,
                    collect_frames = False,
                )
            except Exception:
                # Print full traceback so the cause is visible in the SLURM
                # log even when the job is launched with set -e in run.sh.
                print(f"\n[ERROR] Episode {episode} crashed with:\n"
                      f"{traceback.format_exc()}", flush=True)
                raise

            m = compute_metrics(frames)
            agent.return_history.append(m["total_team_reward"])

            ep_elapsed = time.time() - ep_t0
            last_loss  = agent.loss_history[-1] if agent.loss_history else 0.0

            summary = (
                f"Ep {episode:>4}/{total_eps} | "
                f"steps={m['steps']:>3} | "
                f"R_A={m['total_reward_a']:>7.2f} | "
                f"R_B={m['total_reward_b']:>7.2f} | "
                f"team={m['total_team_reward']:>8.2f} | "
                f"catches={m['n_catches']:>2} | "
                f"maulings={m['n_maulings']:>2} | "
                f"eps={agent.epsilon:.4f} | "
                f"loss={last_loss:.4f} | "
                f"ep_time={ep_elapsed:.1f}s"
            )

            if TQDM_AVAILABLE:
                ep_iter.set_postfix({
                    "team":    f"{m['total_team_reward']:.1f}",
                    "catches": m["n_catches"],
                    "eps":     f"{agent.epsilon:.3f}",
                    "loss":    f"{last_loss:.4f}",
                    "ep_time": f"{ep_elapsed:.1f}s",
                })
                tqdm.write(summary)
            else:
                print(summary, flush=True)

            if episode % CONFIG["checkpoint_every"] == 0:
                agent.save(str(ckpt_dir / f"dqn_ep{episode}.pt"))
                agent.save(str(ckpt_latest))

                save_rollout_csv(
                    multiagent  = True,
                    frames      = frames,
                    output_path = str(ckpt_dir / f"rollout_ep{episode}.csv"),
                )

                window  = min(CONFIG["checkpoint_every"], episode)
                hist    = agent.return_history[-window:]
                rolling = (
                    f"\n  --- checkpoint ep {episode} | last {window} eps | "
                    f"avg team_R = {sum(hist) / window:.2f} ---\n"
                )
                if TQDM_AVAILABLE:
                    tqdm.write(rolling)
                else:
                    print(rolling, flush=True)

    finally:
        # Always save and close cleanly, even if training is interrupted.
        print("\nSaving final checkpoint...")
        agent.save(str(ckpt_dir / "dqn_final.pt"))
        agent.save(str(ckpt_latest))
        env.close()
        print("Training complete.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(checkpoint: str):
    print("=" * 65)
    print("  EVALUATION: Shared DQN (CNN policy) on Stag Hunt")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent  = DQNAgent(
        lr                 = CONFIG["lr"],
        gamma              = CONFIG["gamma"],
        epsilon_start      = 0.0,
        epsilon_end        = 0.0,
        epsilon_decay      = 1.0,
        target_update_freq = CONFIG["target_update_freq"],
        batch_size         = CONFIG["batch_size"],
        buffer_size        = CONFIG["buffer_size"],
        device             = device,
    )
    agent.load(checkpoint)

    total_catches  = 0
    total_maulings = 0
    total_r_a      = 0.0
    total_r_b      = 0.0
    total_steps    = 0

    n        = CONFIG["eval_episodes"]

    # For eval ep 1 we want a video, which requires load_renderer=True.
    # Create a renderer env for that one episode, then switch to a plain env.
    print("Creating environment...")
    env_render = make_env(load_renderer=True)
    env_plain  = make_env(load_renderer=False)
    print("Environment ready.\n")

    ep_range = range(1, n + 1)
    ep_iter  = (
        tqdm(ep_range, desc="Eval episodes", unit="ep", dynamic_ncols=True)
        if TQDM_AVAILABLE else ep_range
    )

    try:
        for ep in ep_iter:
            want_video = CONFIG["save_video"] and ep == 1
            env        = env_render if want_video else env_plain

            try:
                frames = run_episode(
                    agent          = agent,
                    env            = env,
                    episode_idx    = ep,
                    training       = False,
                    collect_frames = want_video,
                )
            except Exception:
                print(f"\n[ERROR] Eval episode {ep} crashed with:\n"
                      f"{traceback.format_exc()}", flush=True)
                raise

            m = compute_metrics(frames)
            total_catches  += m["n_catches"]
            total_maulings += m["n_maulings"]
            total_r_a      += m["total_reward_a"]
            total_r_b      += m["total_reward_b"]
            total_steps    += m["steps"]

            if want_video:
                video_path = f"eval_ep{ep}.mp4"
                save_rollout_video(frames, output_path=video_path, fps=4)
                print(f"  Video saved -> {video_path}")

            save_rollout_csv(
                multiagent  = True,
                frames      = frames,
                output_path = f"eval_ep{ep}.csv",
            )

            summary = (
                f"Eval {ep:>3}/{n} | "
                f"steps={m['steps']:>3} | "
                f"R_A={m['total_reward_a']:>7.2f} | "
                f"R_B={m['total_reward_b']:>7.2f} | "
                f"team={m['total_team_reward']:>8.2f} | "
                f"catches={m['n_catches']:>2} | "
                f"maulings={m['n_maulings']:>2}"
            )
            if TQDM_AVAILABLE:
                ep_iter.set_postfix({
                    "team":    f"{m['total_team_reward']:.1f}",
                    "catches": m["n_catches"],
                })
                tqdm.write(summary)
            else:
                print(summary, flush=True)

    finally:
        env_render.close()
        env_plain.close()

    print(f"\n{'=' * 65}")
    print(f"  RESULTS over {n} episodes")
    print(f"{'=' * 65}")
    print(f"  Avg reward A:          {total_r_a / n:.2f}")
    print(f"  Avg reward B:          {total_r_b / n:.2f}")
    print(f"  Avg team reward:       {(total_r_a + total_r_b) / n:.2f}")
    print(f"  Total catches:         {total_catches}")
    print(f"  Catch rate (per step): {100 * total_catches / max(total_steps, 1):.2f}%")
    print(f"  Total maulings:        {total_maulings}")
    print(f"{'=' * 65}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DQN training/eval for Stag Hunt"
    )
    parser.add_argument(
        "--mode", choices=["train", "eval"], default="train",
        help="train or eval",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/dqn_latest.pt",
        help="Path to checkpoint for eval (or resume point for train)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        evaluate(args.checkpoint)
