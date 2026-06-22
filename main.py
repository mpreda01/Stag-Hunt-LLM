"""
main.py

Training and evaluation entry point for LLM+REINFORCE agents on Stag Hunt.

Usage:
    python main.py --mode train
    python main.py --mode train --prompt_type 4        # two-shot prompt
    python main.py --mode eval  --checkpoint_a checkpoints/agent_A_latest.pt \
                                --checkpoint_b checkpoints/agent_B_latest.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from utils.const import ENV_FACTORIES
from utils.utils import (
    RolloutFrame,
    get_pixel_frame,
    save_rollout_video,
    save_rollout_csv,
)
from agents.qwen4b import obs_to_prompt          # same prompt builder used everywhere
from agents.llm_policy_agent import LLMEncoder, REINFORCEAgent


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

    # Prompt type passed to obs_to_prompt():
    #   "2" = zero-shot, "3" = one-shot, "4" = two-shot
    "prompt_type":        "4",

    # Training
    "total_episodes":     500,
    "reward_shaping":     True,
    "shaping_coeff":      0.1,

    # REINFORCE
    "gamma":              0.99,
    "lr":                 1e-3,

    # Checkpointing
    "checkpoint_dir":     "checkpoints",
    "checkpoint_every":   50,

    # Evaluation
    "eval_episodes":      20,
    "save_video":         True,
}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(load_renderer: bool = False):
    return ENV_FACTORIES[CONFIG["env_name"]](
        obs_type="coords",
        load_renderer=load_renderer,
        enable_multiagent=True,
        grid_size=CONFIG["grid_size"],
        max_timesteps=CONFIG["max_timesteps"],
        stag_reward=CONFIG["stag_reward"],
        forage_reward=CONFIG["forage_reward"],
        mauling_punishment=CONFIG["mauling_punishment"],
    )


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    agent_a: REINFORCEAgent,
    agent_b: REINFORCEAgent,
    training: bool = True,
    save_video_path: str | None = None,
) -> list[RolloutFrame]:
    """
    Run one full episode with both agents.
    Returns the list of RolloutFrames for the episode.

    RolloutFrame stores at each step:
        - step:        timestep index
        - obs:         raw coords observation (2, 10)
        - actions:     [action_a, action_b] ints
        - rewards:     (raw_r_a, raw_r_b) floats  <- always raw env reward
        - pixel_frame: RGB frame for video
        - info:        env info dict

    Training mode:
        - Stochastic action selection
        - Shaped reward stored for REINFORCE (not saved in RolloutFrame)
        - agent.update() called at end of episode

    Evaluation mode:
        - Greedy action selection
        - No update
    """
    load_renderer = save_video_path is not None
    env = make_env(load_renderer=load_renderer)
    obs, info = env.reset()

    frames: list[RolloutFrame] = []

    # Step 0: initial state before any action
    frames.append(RolloutFrame(
        step=0,
        obs=obs,
        actions=None,
        rewards=None,
        pixel_frame=get_pixel_frame(env, multiagent=True) if load_renderer else None,
        info=info,
    ))

    for step in range(1, CONFIG["max_timesteps"] + 1):

        # --- Build prompts from current obs (same as frozen LLM evaluation) ---
        # obs shape: (2, 10) — row 0 = Agent A view, row 1 = Agent B view
        prompt_a, prompt_b = obs_to_prompt(obs, prompot_type=CONFIG["prompt_type"])

        # --- Encode via frozen Qwen -> (2048,) hidden states ---
        hidden_a = agent_a.encode(prompt_a)
        hidden_b = agent_b.encode(prompt_b)

        # --- Select actions ---
        action_a = agent_a.select_action(hidden_a, greedy=not training)
        action_b = agent_b.select_action(hidden_b, greedy=not training)

        # --- Step environment ---
        next_obs, rewards, terminated, truncated, info = env.step([action_a, action_b])
        raw_r_a, raw_r_b = float(rewards[0]), float(rewards[1])

        # --- Reward shaping for REINFORCE (training only) ---
        # RolloutFrame always stores RAW env rewards for clean logging/CSV
        if training:
            store_r_a = REINFORCEAgent.shaped_reward(
                obs[0], next_obs[0], raw_r_a, CONFIG["shaping_coeff"]
            ) if CONFIG["reward_shaping"] else raw_r_a

            store_r_b = REINFORCEAgent.shaped_reward(
                obs[1], next_obs[1], raw_r_b, CONFIG["shaping_coeff"]
            ) if CONFIG["reward_shaping"] else raw_r_b

            agent_a.store_reward(store_r_a)
            agent_b.store_reward(store_r_b)

        # --- Store frame with raw rewards ---
        frames.append(RolloutFrame(
            step=step,
            obs=next_obs,
            actions=[action_a, action_b],
            rewards=(raw_r_a, raw_r_b),
            pixel_frame=get_pixel_frame(env, multiagent=True) if load_renderer else None,
            info=info,
        ))

        obs = next_obs

        if terminated or truncated:
            break

    # --- REINFORCE update at end of episode ---
    if training:
        agent_a.update()
        agent_b.update()

    # --- Save video BEFORE env is garbage collected (avoids pygame quit() crash) ---
    if save_video_path and load_renderer:
        save_rollout_video(frames, output_path=save_video_path, fps=4)
        print(f"  Video saved -> {save_video_path}")

    return frames


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(frames: list[RolloutFrame]) -> dict:
    """Extract episode statistics from a list of RolloutFrames."""
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
        "total_reward_a": total_r_a,
        "total_reward_b": total_r_b,
        "n_catches":      n_catches,
        "n_maulings":     n_maulings,
        "steps":          len(frames) - 1,   # exclude step-0 frame
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(prompt_type: str | None = None):
    if prompt_type:
        CONFIG["prompt_type"] = prompt_type

    print("=" * 65)
    print(f"  TRAINING: LLM+REINFORCE | prompt_type={CONFIG['prompt_type']}")
    print("=" * 65)

    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # One shared frozen encoder, two independent policy heads
    encoder = LLMEncoder(device=device)
    agent_a = REINFORCEAgent(encoder, agent_id="A", lr=CONFIG["lr"], gamma=CONFIG["gamma"])
    agent_b = REINFORCEAgent(encoder, agent_id="B", lr=CONFIG["lr"], gamma=CONFIG["gamma"])

    # Resume from checkpoint if available
    ckpt_a = ckpt_dir / "agent_A_latest.pt"
    ckpt_b = ckpt_dir / "agent_B_latest.pt"
    start_episode = 1
    if ckpt_a.exists() and ckpt_b.exists():
        agent_a.load(str(ckpt_a))
        agent_b.load(str(ckpt_b))
        start_episode = len(agent_a.episode_return_history) + 1
        print(f"Resuming from episode {start_episode}\n")

    all_frames: list[list[RolloutFrame]] = []

    for episode in range(start_episode, CONFIG["total_episodes"] + 1):
        t0 = time.time()

        frames  = run_episode(agent_a, agent_b, training=True)
        metrics = compute_metrics(frames)
        all_frames.append(frames)

        loss_a = agent_a.loss_history[-1] if agent_a.loss_history else 0.0
        loss_b = agent_b.loss_history[-1] if agent_b.loss_history else 0.0

        print(
            f"Ep {episode:>4}/{CONFIG['total_episodes']} | "
            f"Steps: {metrics['steps']:>3} | "
            f"R_A: {metrics['total_reward_a']:>7.2f} | "
            f"R_B: {metrics['total_reward_b']:>7.2f} | "
            f"Catches: {metrics['n_catches']:>2} | "
            f"Maulings: {metrics['n_maulings']:>2} | "
            f"Loss_A: {loss_a:>8.3f} | "
            f"Loss_B: {loss_b:>8.3f} | "
            f"Time: {time.time()-t0:.1f}s"
        )

        # Checkpoint + save CSV of last episode
        if episode % CONFIG["checkpoint_every"] == 0:
            agent_a.save(str(ckpt_dir / f"agent_A_ep{episode}.pt"))
            agent_b.save(str(ckpt_dir / f"agent_B_ep{episode}.pt"))
            agent_a.save(str(ckpt_a))
            agent_b.save(str(ckpt_b))

            save_rollout_csv(
                multiagent=True,
                frames=frames,
                output_path=str(ckpt_dir / f"rollout_ep{episode}.csv"),
            )

            window  = min(CONFIG["checkpoint_every"], episode)
            hist_a  = agent_a.episode_return_history[-window:]
            hist_b  = agent_b.episode_return_history[-window:]
            print(
                f"\n  --- Last {window} eps | "
                f"Avg R_A: {sum(hist_a)/window:.2f} | "
                f"Avg R_B: {sum(hist_b)/window:.2f} ---\n"
            )

    agent_a.save(str(ckpt_dir / "agent_A_final.pt"))
    agent_b.save(str(ckpt_dir / "agent_B_final.pt"))
    print("\nTraining complete.")
    return all_frames


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(checkpoint_a: str, checkpoint_b: str):
    print("=" * 65)
    print("  EVALUATION: LLM+REINFORCE on Stag Hunt")
    print("=" * 65)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = LLMEncoder(device=device)
    agent_a = REINFORCEAgent(encoder, agent_id="A", lr=CONFIG["lr"], gamma=CONFIG["gamma"])
    agent_b = REINFORCEAgent(encoder, agent_id="B", lr=CONFIG["lr"], gamma=CONFIG["gamma"])
    agent_a.load(checkpoint_a)
    agent_b.load(checkpoint_b)

    total_catches  = 0
    total_maulings = 0
    total_r_a      = 0.0
    total_r_b      = 0.0
    total_steps    = 0

    for ep in range(1, CONFIG["eval_episodes"] + 1):
        # Save video only for episode 1
        video_path = f"eval_ep{ep}.mp4" if (CONFIG["save_video"] and ep == 1) else None

        frames  = run_episode(agent_a, agent_b, training=False, save_video_path=video_path)
        metrics = compute_metrics(frames)

        total_catches  += metrics["n_catches"]
        total_maulings += metrics["n_maulings"]
        total_r_a      += metrics["total_reward_a"]
        total_r_b      += metrics["total_reward_b"]
        total_steps    += metrics["steps"]

        # Save CSV for every eval episode
        save_rollout_csv(
            multiagent=True,
            frames=frames,
            output_path=f"eval_ep{ep}.csv",
        )

        print(
            f"Eval {ep:>3} | "
            f"Steps: {metrics['steps']:>3} | "
            f"R_A: {metrics['total_reward_a']:>7.2f} | "
            f"R_B: {metrics['total_reward_b']:>7.2f} | "
            f"Catches: {metrics['n_catches']:>2} | "
            f"Maulings: {metrics['n_maulings']:>2}"
        )

    n = CONFIG["eval_episodes"]
    print(f"\n{'='*65}")
    print(f"  RESULTS over {n} episodes")
    print(f"{'='*65}")
    print(f"  Avg reward A:          {total_r_a/n:.2f}")
    print(f"  Avg reward B:          {total_r_b/n:.2f}")
    print(f"  Avg combined reward:   {(total_r_a+total_r_b)/n:.2f}")
    print(f"  Total catches:         {total_catches}")
    print(f"  Catch rate (per step): {100*total_catches/total_steps:.2f}%")
    print(f"  Total maulings:        {total_maulings}")
    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",          choices=["train", "eval"], default="train")
    parser.add_argument("--prompt_type",   choices=["2", "3", "4"],   default="4",
                        help="2=zero-shot, 3=one-shot, 4=two-shot")
    parser.add_argument("--checkpoint_a",  default="checkpoints/agent_A_latest.pt")
    parser.add_argument("--checkpoint_b",  default="checkpoints/agent_B_latest.pt")
    args = parser.parse_args()

    if args.mode == "train":
        train(prompt_type=args.prompt_type)
    else:
        evaluate(args.checkpoint_a, args.checkpoint_b)
