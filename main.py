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

W&B integration:
    Every episode logs per-episode metrics (rewards, catches, maulings,
    epsilon, loss, replay buffer size, episode duration).
    Every checkpoint_every episodes also logs a rolling-window average.
    wandb.watch() is called on the online CNN so gradient/weight
    histograms appear in the W&B UI automatically.
    On eval, a summary table and (optionally) the eval video are uploaded.

Usage:
    python main.py --mode train
    python main.py --mode eval --checkpoint checkpoints/dqn_latest.pt

    W&B is enabled by default. Disable with --no-wandb (e.g. quick local
    debugging runs).  Set WANDB_API_KEY in your environment before running.
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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[W&B] wandb not installed — logging disabled. Run: pip install wandb")

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

    # W&B
    "wandb_project": "stag-hunt-dqn",
    "wandb_entity":  None,   # set to your W&B username / team name, or leave None
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

def train(use_wandb: bool = True):
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
    wandb_id_file = ckpt_dir / "wandb_run_id.txt"   # persists the run ID across SLURM jobs
    start_episode = 1
    if ckpt_latest.exists():
        agent.load(str(ckpt_latest))
        start_episode = len(agent.return_history) + 1
        print(f"Resuming from episode {start_episode}\n")

    total_eps = CONFIG["total_episodes"]

    # ------------------------------------------------------------------
    # W&B initialisation
    #
    # Key: we persist the wandb run ID to disk alongside the checkpoint.
    # On resume, we pass that same ID back so W&B appends to the existing
    # run instead of creating a new one that restarts the x-axis at 0.
    # ------------------------------------------------------------------
    run = None
    if use_wandb and WANDB_AVAILABLE:
        # Recover the run ID from a previous job if it exists
        existing_run_id = None
        if wandb_id_file.exists():
            existing_run_id = wandb_id_file.read_text().strip()
            print(f"[W&B] Resuming run ID: {existing_run_id}")

        run = wandb.init(
            project = CONFIG["wandb_project"],
            entity  = CONFIG["wandb_entity"],
            # Keep a stable human-readable name; only set it on the first run
            # so resumed runs don't get a new timestamped name.
            name    = f"dqn-hunt" if existing_run_id else f"dqn-hunt-{time.strftime('%Y%m%d-%H%M%S')}",
            config  = CONFIG,
            id      = existing_run_id,   # None on first run → W&B auto-generates one
            resume  = "must" if existing_run_id else "never",
            tags    = ["dqn", "hunt", "image-obs", "shared-buffer"],
        )

        # Persist the run ID so the next SLURM job can resume correctly
        wandb_id_file.write_text(run.id)

        # Log gradient norms and weight histograms every 100 update steps.
        # log_freq is in gradient-update steps, not episodes.
        wandb.watch(agent.online, log="all", log_freq=100)
        print(f"[W&B] Run URL: {run.url}\n")
    else:
        print("[W&B] Logging disabled for this run.\n")

    # Create the environment once and reuse it for the entire training run.
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
                    agent          = agent,
                    env            = env,
                    episode_idx    = episode,
                    training       = True,
                    collect_frames = False,
                )
            except Exception:
                print(f"\n[ERROR] Episode {episode} crashed with:\n"
                      f"{traceback.format_exc()}", flush=True)
                raise

            m = compute_metrics(frames)
            agent.return_history.append(m["total_team_reward"])

            ep_elapsed = time.time() - ep_t0
            last_loss  = agent.loss_history[-1] if agent.loss_history else 0.0

            # ----------------------------------------------------------------
            # W&B — per-episode metrics
            # ----------------------------------------------------------------
            if run is not None:
                wandb.log({
                    # Rewards
                    "episode/reward_A":          m["total_reward_a"],
                    "episode/reward_B":          m["total_reward_b"],
                    "episode/team_reward":       m["total_team_reward"],
                    # Cooperation signals
                    "episode/n_catches":         m["n_catches"],
                    "episode/n_maulings":        m["n_maulings"],
                    "episode/catch_rate":        m["n_catches"] / max(m["steps"], 1),
                    "episode/maul_rate":         m["n_maulings"] / max(m["steps"], 1),
                    # Training diagnostics
                    "train/epsilon":             agent.epsilon,
                    "train/loss":                last_loss,
                    "train/replay_buffer_size":  len(agent.buffer),
                    "train/update_count":        agent._update_count,
                    # Timing
                    "train/episode_duration_s":  ep_elapsed,
                    "train/steps_per_episode":   m["steps"],
                }, step=episode)

            # ----------------------------------------------------------------
            # Console summary
            # ----------------------------------------------------------------
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

            # ----------------------------------------------------------------
            # Checkpoint + W&B rolling-window summary
            # ----------------------------------------------------------------
            if episode % CONFIG["checkpoint_every"] == 0:
                ckpt_path = str(ckpt_dir / f"dqn_ep{episode}.pt")
                agent.save(ckpt_path)
                agent.save(str(ckpt_latest))

                save_rollout_csv(
                    multiagent  = True,
                    frames      = frames,
                    output_path = str(ckpt_dir / f"rollout_ep{episode}.csv"),
                )

                window  = min(CONFIG["checkpoint_every"], episode)
                hist    = agent.return_history[-window:]
                avg_r   = sum(hist) / window

                rolling = (
                    f"\n  --- checkpoint ep {episode} | last {window} eps | "
                    f"avg team_R = {avg_r:.2f} ---\n"
                )
                if TQDM_AVAILABLE:
                    tqdm.write(rolling)
                else:
                    print(rolling, flush=True)

                if run is not None:
                    # Rolling-window averages logged at checkpoint frequency
                    recent = agent.loss_history[-window * CONFIG["max_timesteps"]:]
                    avg_loss = sum(recent) / len(recent) if recent else 0.0

                    wandb.log({
                        f"checkpoint/avg_team_reward_last_{window}ep": avg_r,
                        f"checkpoint/avg_loss_last_{window}ep":        avg_loss,
                    }, step=episode)

                    # Save checkpoint as a W&B artifact so it can be
                    # downloaded and resumed later without SSH access.
                    artifact = wandb.Artifact(
                        name = f"dqn-checkpoint-ep{episode}",
                        type = "model",
                        metadata = {
                            "episode":       episode,
                            "avg_team_reward": avg_r,
                            "epsilon":       agent.epsilon,
                        },
                    )
                    artifact.add_file(ckpt_path)
                    run.log_artifact(artifact)

    finally:
        # Always save and close cleanly, even if training is interrupted.
        print("\nSaving final checkpoint...")
        final_path = str(ckpt_dir / "dqn_final.pt")
        agent.save(final_path)
        agent.save(str(ckpt_latest))
        env.close()

        if run is not None:
            # Upload the final model as a "best / latest" artifact.
            artifact = wandb.Artifact(
                name = "dqn-final",
                type = "model",
                metadata = {"total_episodes": total_eps},
            )
            artifact.add_file(final_path)
            run.log_artifact(artifact)
            wandb.finish()

        print("Training complete.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(checkpoint: str, use_wandb: bool = True):
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

    # ------------------------------------------------------------------
    # W&B initialisation for eval
    # ------------------------------------------------------------------
    run = None
    if use_wandb and WANDB_AVAILABLE:
        run = wandb.init(
            project = CONFIG["wandb_project"],
            entity  = CONFIG["wandb_entity"],
            name    = f"eval-{Path(checkpoint).stem}-{time.strftime('%Y%m%d-%H%M%S')}",
            config  = {**CONFIG, "checkpoint": checkpoint, "mode": "eval"},
            tags    = ["eval", "dqn", "hunt"],
        )
        print(f"[W&B] Run URL: {run.url}\n")

    total_catches  = 0
    total_maulings = 0
    total_r_a      = 0.0
    total_r_b      = 0.0
    total_steps    = 0

    n = CONFIG["eval_episodes"]

    # W&B Table to log per-episode eval results in a sortable UI widget
    eval_table = wandb.Table(
        columns=["episode", "steps", "reward_A", "reward_B",
                 "team_reward", "catches", "maulings", "catch_rate"]
    ) if run is not None else None

    # For eval ep 1 we want a video, which requires load_renderer=True.
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

                # Upload the video to W&B so you can watch it in the browser
                if run is not None:
                    wandb.log({"eval/rollout_video": wandb.Video(video_path, fps=4, format="mp4")}, step=ep)

            save_rollout_csv(
                multiagent  = True,
                frames      = frames,
                output_path = f"eval_ep{ep}.csv",
            )

            catch_rate = m["n_catches"] / max(m["steps"], 1)

            # Per-episode W&B metrics
            if run is not None:
                wandb.log({
                    "eval/reward_A":    m["total_reward_a"],
                    "eval/reward_B":    m["total_reward_b"],
                    "eval/team_reward": m["total_team_reward"],
                    "eval/catches":     m["n_catches"],
                    "eval/maulings":    m["n_maulings"],
                    "eval/catch_rate":  catch_rate,
                    "eval/steps":       m["steps"],
                }, step=ep)

                eval_table.add_data(
                    ep,
                    m["steps"],
                    m["total_reward_a"],
                    m["total_reward_b"],
                    m["total_team_reward"],
                    m["n_catches"],
                    m["n_maulings"],
                    catch_rate,
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

    # Aggregate results
    avg_r_a    = total_r_a / n
    avg_r_b    = total_r_b / n
    avg_team_r = (total_r_a + total_r_b) / n
    catch_rate = 100 * total_catches / max(total_steps, 1)

    print(f"\n{'=' * 65}")
    print(f"  RESULTS over {n} episodes")
    print(f"{'=' * 65}")
    print(f"  Avg reward A:          {avg_r_a:.2f}")
    print(f"  Avg reward B:          {avg_r_b:.2f}")
    print(f"  Avg team reward:       {avg_team_r:.2f}")
    print(f"  Total catches:         {total_catches}")
    print(f"  Catch rate (per step): {catch_rate:.2f}%")
    print(f"  Total maulings:        {total_maulings}")
    print(f"{'=' * 65}")

    # Upload aggregate summary + table to W&B
    if run is not None:
        wandb.log({
            "eval_summary/avg_reward_A":          avg_r_a,
            "eval_summary/avg_reward_B":          avg_r_b,
            "eval_summary/avg_team_reward":       avg_team_r,
            "eval_summary/total_catches":         total_catches,
            "eval_summary/catch_rate_pct":        catch_rate,
            "eval_summary/total_maulings":        total_maulings,
            "eval_summary/per_episode_results":   eval_table,
        })
        wandb.finish()


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
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging (useful for quick local tests)",
    )
    args = parser.parse_args()

    use_wandb = not args.no_wandb

    if args.mode == "train":
        train(use_wandb=use_wandb)
    else:
        evaluate(args.checkpoint, use_wandb=use_wandb)
