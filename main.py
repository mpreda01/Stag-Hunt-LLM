"""
main.py  —  LoRA + PPO fine-tuning of Qwen3-4B on Stag Hunt

Architecture:
    obs -> two-shot prompt
        -> Qwen3-4B + LoRA (trainable attention adapters)
        -> sample action from action-token logits
        -> env.step() -> team reward
        -> PPO update on LoRA weights + value head

Usage:
    python main.py --mode train
    python main.py --mode train --prompt_type 4
    python main.py --mode eval  --checkpoint checkpoints/policy_ep200

Target hardware: L40 (48 GB VRAM)  —  set --partition=l40 in run.sh
"""

import argparse
import time
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

import torch

from utils.const import ENV_FACTORIES
from utils.utils import (
    RolloutFrame,
    get_pixel_frame,
    save_rollout_video,
    save_rollout_csv,
)
from agents.qwen4b import obs_to_prompt
from agents.llm_policy_agent import QwenLoRAPolicy, PPOAgent

import os
os.environ["HF_HOME"]            = "/scratch.hpc/matteo.preda/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch.hpc/matteo.preda/hf_cache"


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

    # Prompt
    "prompt_type":        "4",   # 2=zero-shot  3=one-shot  4=two-shot

    # Training
    "total_episodes":     500,

    # PPO hyperparameters
    "lr":        1e-4,    # AdamW lr for LoRA params
    "gamma":     0.99,    # discount factor
    "clip_eps":  0.2,     # PPO clip ratio
    "vf_coeff":  0.5,     # value loss coefficient
    "ent_coeff": 0.01,    # entropy bonus coefficient

    # LoRA
    "lora_rank":    16,
    "lora_alpha":   32,
    "lora_dropout": 0.05,

    # Checkpointing
    "checkpoint_dir":   "checkpoints",
    "checkpoint_every": 50,

    # Evaluation
    "eval_episodes": 20,
    "save_video":    True,
}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(load_renderer: bool = False):
    return ENV_FACTORIES[CONFIG["env_name"]](
        obs_type        = "coords",
        load_renderer   = load_renderer,
        enable_multiagent = True,
        grid_size       = CONFIG["grid_size"],
        max_timesteps   = CONFIG["max_timesteps"],
        stag_reward     = CONFIG["stag_reward"],
        forage_reward   = CONFIG["forage_reward"],
        mauling_punishment = CONFIG["mauling_punishment"],
    )


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    agent_a: PPOAgent,
    agent_b: PPOAgent,
    episode_idx: int,
    training:    bool = True,
    save_video_path: str | None = None,
) -> list[RolloutFrame]:
    """
    Run one full episode with both PPO agents.

    Reward:
        team_reward = raw_r_a + raw_r_b  (shared, no shaping)
        Both agents receive the same team reward — each agent cares
        about its partner's outcome without being told how to cooperate.

    Both agents share the same QwenLoRAPolicy so both sets of
    (log_prob, value, entropy) computed in this episode contribute
    to the same PPO update — doubling the effective batch size.
    """
    load_renderer = save_video_path is not None
    env = make_env(load_renderer=load_renderer)
    obs, info = env.reset()

    frames: list[RolloutFrame] = []
    frames.append(RolloutFrame(
        step        = 0,
        obs         = obs,
        actions     = None,
        rewards     = None,
        pixel_frame = get_pixel_frame(env, multiagent=True) if load_renderer else None,
        info        = info,
    ))

    mode_str = "Train" if training else "Eval"

    if TQDM_AVAILABLE:
        step_iter = tqdm(
            range(1, CONFIG["max_timesteps"] + 1),
            desc         = f"  {mode_str} ep {episode_idx} steps",
            unit         = "step",
            leave        = False,
            dynamic_ncols = True,
        )
    else:
        step_iter = range(1, CONFIG["max_timesteps"] + 1)

    for step in step_iter:
        step_t0 = time.time()

        prompt_a, prompt_b = obs_to_prompt(obs, prompot_type=CONFIG["prompt_type"])

        action_a = agent_a.select_action(prompt_a, greedy=not training)
        action_b = agent_b.select_action(prompt_b, greedy=not training)

        next_obs, rewards, terminated, truncated, info = env.step([action_a, action_b])
        raw_r_a, raw_r_b = float(rewards[0]), float(rewards[1])

        if training:
            team_reward = raw_r_a + raw_r_b
            agent_a.store_reward(team_reward)
            agent_b.store_reward(team_reward)

        frames.append(RolloutFrame(
            step        = step,
            obs         = next_obs,
            actions     = [action_a, action_b],
            rewards     = (raw_r_a, raw_r_b),
            pixel_frame = get_pixel_frame(env, multiagent=True) if load_renderer else None,
            info        = info,
        ))

        step_elapsed = time.time() - step_t0

        if TQDM_AVAILABLE:
            step_iter.set_postfix({
                "r_A":    f"{raw_r_a:+.1f}",
                "r_B":    f"{raw_r_b:+.1f}",
                "team":   f"{raw_r_a+raw_r_b:+.1f}",
                "act":    f"{action_a},{action_b}",
                "s/step": f"{step_elapsed:.1f}s",
            })
        else:
            print(
                f"  [{mode_str}] ep {episode_idx} | step {step:>3} | "
                f"r_A={raw_r_a:+.1f} r_B={raw_r_b:+.1f} "
                f"team={raw_r_a+raw_r_b:+.1f} | "
                f"acts=({action_a},{action_b}) | {step_elapsed:.2f}s"
            )

        obs = next_obs
        if terminated or truncated:
            break

    # PPO update — one per episode, after all steps collected
    if training:
        loss_a = agent_a.update()
        loss_b = agent_b.update()
    
    if save_video_path and load_renderer:
        save_rollout_video(frames, output_path=save_video_path, fps=4)
        print(f"  Video saved -> {save_video_path}")

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

def train(prompt_type: str | None = None):
    if prompt_type:
        CONFIG["prompt_type"] = prompt_type

    print("=" * 65)
    print(f"  TRAINING: Qwen3-4B LoRA + PPO on Stag Hunt")
    print(f"  prompt_type={CONFIG['prompt_type']} | lr={CONFIG['lr']}")
    print(f"  lora_rank={CONFIG['lora_rank']} | clip_eps={CONFIG['clip_eps']}")
    print("=" * 65)

    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # Both agents share one policy (same LoRA weights, different rollout buffers)
    policy  = QwenLoRAPolicy(
        device       = device,
        lora_rank    = CONFIG["lora_rank"],
        lora_alpha   = CONFIG["lora_alpha"],
        lora_dropout = CONFIG["lora_dropout"],
    )
    agent_a = PPOAgent(
        policy    = policy,
        agent_id  = "A",
        lr        = CONFIG["lr"],
        gamma     = CONFIG["gamma"],
        clip_eps  = CONFIG["clip_eps"],
        vf_coeff  = CONFIG["vf_coeff"],
        ent_coeff = CONFIG["ent_coeff"],
    )
    agent_b = PPOAgent(
        policy    = policy,
        agent_id  = "B",
        lr        = CONFIG["lr"],
        gamma     = CONFIG["gamma"],
        clip_eps  = CONFIG["clip_eps"],
        vf_coeff  = CONFIG["vf_coeff"],
        ent_coeff = CONFIG["ent_coeff"],
    )

    # Resume from checkpoint if available
    ckpt_policy = str(ckpt_dir / "policy_latest")
    ckpt_a_opt  = str(ckpt_dir / "agent_A_latest")
    ckpt_b_opt  = str(ckpt_dir / "agent_B_latest")
    start_episode = 1

    if Path(ckpt_policy + "_lora").exists():
        policy.load(ckpt_policy)
        agent_a.load(ckpt_a_opt)
        agent_b.load(ckpt_b_opt)
        start_episode = len(agent_a.episode_return_history) + 1
        print(f"Resuming from episode {start_episode}\n")

    total_eps = CONFIG["total_episodes"]

    if TQDM_AVAILABLE:
        ep_bar = tqdm(
            range(start_episode, total_eps + 1),
            desc          = "Training episodes",
            unit          = "ep",
            dynamic_ncols = True,
        )
    else:
        ep_bar = range(start_episode, total_eps + 1)

    for episode in ep_bar:
        ep_t0 = time.time()

        frames     = run_episode(agent_a, agent_b, episode_idx=episode, training=True)
        metrics    = compute_metrics(frames)
        ep_elapsed = time.time() - ep_t0

        loss_a = agent_a.loss_history[-1] if agent_a.loss_history else 0.0
        loss_b = agent_b.loss_history[-1] if agent_b.loss_history else 0.0

        summary = (
            f"Ep {episode:>4}/{total_eps} | "
            f"steps={metrics['steps']:>3} | "
            f"R_A={metrics['total_reward_a']:>7.2f} | "
            f"R_B={metrics['total_reward_b']:>7.2f} | "
            f"team={metrics['total_team_reward']:>8.2f} | "
            f"catches={metrics['n_catches']:>2} | "
            f"maulings={metrics['n_maulings']:>2} | "
            f"loss_A={loss_a:>8.4f} | "
            f"loss_B={loss_b:>8.4f} | "
            f"ep_time={ep_elapsed:.1f}s"
        )

        if TQDM_AVAILABLE:
            ep_bar.set_postfix({
                "team":     f"{metrics['total_team_reward']:.1f}",
                "catches":  metrics["n_catches"],
                "loss_A":   f"{loss_a:.4f}",
                "ep_time":  f"{ep_elapsed:.1f}s",
            })
            tqdm.write(summary)
        else:
            print(summary)

        if episode % CONFIG["checkpoint_every"] == 0:
            ep_tag = str(ckpt_dir / f"policy_ep{episode}")
            policy.save(ep_tag)
            agent_a.save(str(ckpt_dir / f"agent_A_ep{episode}"))
            agent_b.save(str(ckpt_dir / f"agent_B_ep{episode}"))
            policy.save(ckpt_policy)
            agent_a.save(ckpt_a_opt)
            agent_b.save(ckpt_b_opt)

            save_rollout_csv(
                multiagent  = True,
                frames      = frames,
                output_path = str(ckpt_dir / f"rollout_ep{episode}.csv"),
            )

            window = min(CONFIG["checkpoint_every"], episode)
            hist_a = agent_a.episode_return_history[-window:]
            hist_b = agent_b.episode_return_history[-window:]
            rolling = (
                f"\n  --- checkpoint ep {episode} | last {window} eps | "
                f"avg team_R (A)={sum(hist_a)/window:.2f} "
                f"(B)={sum(hist_b)/window:.2f} ---\n"
            )
            if TQDM_AVAILABLE:
                tqdm.write(rolling)
            else:
                print(rolling)

    policy.save(str(ckpt_dir / "policy_final"))
    agent_a.save(str(ckpt_dir / "agent_A_final"))
    agent_b.save(str(ckpt_dir / "agent_B_final"))
    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(checkpoint: str):
    print("=" * 65)
    print("  EVALUATION: Qwen3-4B LoRA + PPO on Stag Hunt")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy  = QwenLoRAPolicy(device=device)
    agent_a = PPOAgent(policy=policy, agent_id="A",
                       lr=CONFIG["lr"], gamma=CONFIG["gamma"])
    agent_b = PPOAgent(policy=policy, agent_id="B",
                       lr=CONFIG["lr"], gamma=CONFIG["gamma"])

    policy.load(checkpoint)
    agent_a.load(checkpoint.replace("policy", "agent_A"))
    agent_b.load(checkpoint.replace("policy", "agent_B"))

    total_catches  = 0
    total_maulings = 0
    total_r_a      = 0.0
    total_r_b      = 0.0
    total_steps    = 0

    n = CONFIG["eval_episodes"]

    if TQDM_AVAILABLE:
        ep_bar = tqdm(range(1, n + 1), desc="Eval episodes",
                      unit="ep", dynamic_ncols=True)
    else:
        ep_bar = range(1, n + 1)

    for ep in ep_bar:
        ep_t0      = time.time()
        video_path = f"eval_ep{ep}.mp4" if (CONFIG["save_video"] and ep == 1) else None
        frames     = run_episode(agent_a, agent_b, episode_idx=ep,
                                 training=False, save_video_path=video_path)
        metrics    = compute_metrics(frames)
        ep_elapsed = time.time() - ep_t0

        total_catches  += metrics["n_catches"]
        total_maulings += metrics["n_maulings"]
        total_r_a      += metrics["total_reward_a"]
        total_r_b      += metrics["total_reward_b"]
        total_steps    += metrics["steps"]

        save_rollout_csv(multiagent=True, frames=frames,
                         output_path=f"eval_ep{ep}.csv")

        summary = (
            f"Eval {ep:>3}/{n} | "
            f"steps={metrics['steps']:>3} | "
            f"R_A={metrics['total_reward_a']:>7.2f} | "
            f"R_B={metrics['total_reward_b']:>7.2f} | "
            f"team={metrics['total_team_reward']:>8.2f} | "
            f"catches={metrics['n_catches']:>2} | "
            f"maulings={metrics['n_maulings']:>2} | "
            f"ep_time={ep_elapsed:.1f}s"
        )
        if TQDM_AVAILABLE:
            ep_bar.set_postfix({
                "team":    f"{metrics['total_team_reward']:.1f}",
                "catches": metrics["n_catches"],
                "ep_time": f"{ep_elapsed:.1f}s",
            })
            tqdm.write(summary)
        else:
            print(summary)

    print(f"\n{'='*65}")
    print(f"  RESULTS over {n} episodes")
    print(f"{'='*65}")
    print(f"  Avg reward A:          {total_r_a/n:.2f}")
    print(f"  Avg reward B:          {total_r_b/n:.2f}")
    print(f"  Avg team reward:       {(total_r_a+total_r_b)/n:.2f}")
    print(f"  Total catches:         {total_catches}")
    print(f"  Catch rate (per step): {100*total_catches/max(total_steps,1):.2f}%")
    print(f"  Total maulings:        {total_maulings}")
    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",         choices=["train", "eval"], default="train")
    parser.add_argument("--prompt_type",  choices=["2", "3", "4"],   default="4",
                        help="2=zero-shot  3=one-shot  4=two-shot")
    parser.add_argument("--checkpoint",   default="checkpoints/policy_latest",
                        help="Path prefix for eval (without _lora suffix)")
    args = parser.parse_args()

    if args.mode == "train":
        train(prompt_type=args.prompt_type)
    else:
        evaluate(args.checkpoint)
