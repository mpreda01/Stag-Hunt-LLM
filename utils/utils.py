import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from utils.const import SYSTEM_PROMPT
from agents.qwen4b import query_llm
import time

@dataclass
class RolloutFrame:
    """A single timestep snapshot from a rollout."""
    step: int
    obs: Any                        # raw agent obs (coords array or image array)
    actions: Any                    # action(s) taken
    rewards: Any                    # reward(s) received
    pixel_frame: np.ndarray         # RGB frame for video, shape (H, W, 3)
    info: dict = field(default_factory=dict)


def get_pixel_frame(env, multiagent: bool) -> np.ndarray:
    """Extract an RGB pixel frame from the renderer."""
    renderer = env.game.RENDERER
    if renderer is None:
        raise RuntimeError(
            "RENDERER is None. Make sure to pass load_renderer=True "
            "or use obs_type='image' when creating the env."
        )
    frame = renderer._update_render(return_observation=True)
    if frame is None:
        raise RuntimeError("_update_render returned None — pygame may not be initialized.")
    if multiagent and len(frame.shape) == 4:
        frame = frame[0]
    return frame


def save_rollout_video(
    frames: list[RolloutFrame],
    output_path: str = "rollout.mp4",
    fps: int = 2,
) -> Path:
    """Save a list of RolloutFrames as an mp4 video.
    
    Works regardless of how the rollout was collected (coords or image obs_type),
    because RolloutFrame always stores the raw pixel frame separately.
    """
    if not frames:
        raise ValueError("frames list is empty.")

    output_path = Path(output_path)
    h, w = frames[0].pixel_frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    try:
        for f in frames:
            writer.write(cv2.cvtColor(f.pixel_frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return output_path

def save_rollout_csv(multiagent: bool, frames: list[RolloutFrame], output_path: str = "rollout.csv") -> Path:
    if multiagent:
        df = pd.DataFrame([{
            "step": f.step,
            "action_a": int(f.actions[0]) if f.actions is not None else None,
            "action_b": int(f.actions[1]) if f.actions is not None else None,
            "reward_a": float(f.rewards[0]) if f.rewards is not None else None,
            "reward_b": float(f.rewards[1]) if f.rewards is not None else None,
            } for f in frames])
    else:
        df = pd.DataFrame([{
            "step": f.step,
            "actions": int(f.actions) if f.actions is not None else None,
            "rewards": float(f.rewards) if f.rewards is not None else None,
        } for f in frames])
    
    df.to_csv(output_path, index=False)
    
def run_random_rollout(policy_fn, env_factory, steps=200, agent_obs_type="coords", **env_kwargs) -> list[RolloutFrame]:
    env = env_factory(obs_type=agent_obs_type, load_renderer=True, **env_kwargs)
    obs, info = env.reset()
    multiagent = bool(getattr(env, "enable_multiagent", True))

    frames = []
    frames.append(RolloutFrame(
        step=0, obs=obs, actions=None, rewards=None,
        pixel_frame=get_pixel_frame(env, multiagent), info=info,
    ))

    for step in range(1, steps + 1):
        actions = policy_fn(obs, env, info)
        obs, rewards, terminated, truncated, info = env.step(actions)
        frames.append(RolloutFrame(
            step=step, obs=obs, actions=actions, rewards=rewards,
            pixel_frame=get_pixel_frame(env, multiagent), info=info,
        ))
        if terminated or truncated:
            break

    return frames




def obs_to_prompt(obs, grid_size=(5, 5)) -> str | tuple[str, str]:
    """
    Convert coords observation to prompt(s).
    - Single agent (obs shape (10,)): pass agent="A" or "B", returns one prompt.
    - Multiagent (obs shape (2,10)): returns (prompt_A, prompt_B) tuple, agent param ignored.
    """
    obs = np.array(obs)

    if obs.ndim == 2:
        # Multiagent: row 0 is A's view, row 1 is B's view
        return (
            build_prompt(obs[0], agent="A", grid_size=grid_size),
            build_prompt(obs[1], agent="B", grid_size=grid_size),
        )
    else:
        # Single agent
        return build_prompt(obs, agent="A", grid_size=grid_size)


def build_prompt(obs, agent: str, grid_size=(5, 5)) -> str:
    """Build a single agent prompt from a flat (10,) obs array."""
    ax, ay = int(obs[0]), int(obs[1])
    bx, by = int(obs[2]), int(obs[3])
    sx, sy = int(obs[4]), int(obs[5])
    plants = [(int(obs[i]), int(obs[i+1])) for i in range(6, len(obs), 2)]
    plants_str = ", ".join(f"({px},{py})" for px, py in plants)

    teammate = "B" if agent == "A" else "A"
    # prova a non ritornare SYSTEM_PROMPT per ogni stato, ma solo la descizione, se llm allucina perchè le istruzioni escono dalla context window manda sempre prompt completo
    return SYSTEM_PROMPT + f"""You are Agent {agent} on a {grid_size[0]}x{grid_size[1]} grid.

                Current positions:
                - You (Agent {agent}): ({ax},{ay})
                - Teammate (Agent {teammate}): ({bx},{by})
                - Stag: ({sx},{sy})
                - Plants: {plants_str}

                The Stag is moving toward the nearest agent. Your teammate is also trying to catch the Stag cooperatively.

                What is your next move? Think about where the Stag will be next turn, and whether your teammate can also reach it.
                ACTION:"""

def run_llm_rollout(
    env_factory,
    steps: int = 200,
    agent_obs_type: str = "coords",
    think: bool = False,
    **env_kwargs
) -> list[RolloutFrame]:
    
    env = env_factory(obs_type=agent_obs_type, load_renderer=True, **env_kwargs)
    obs, info = env.reset()
    multiagent = bool(getattr(env, "enable_multiagent", True))

    frames = []
    frames.append(RolloutFrame(
        step=0, obs=obs, actions=None, rewards=None,
        pixel_frame=get_pixel_frame(env, multiagent), info=info,
    ))

    for step in range(1, steps + 1):
        print(f"\n\n=== Step {step} ===")
        start_time = time.time()
        if multiagent:
            prompt_a, prompt_b = obs_to_prompt(obs)
            action_a, _ = query_llm(prompt_a, think=think)
            action_b, _ = query_llm(prompt_b, think=think)
            actions = [action_a, action_b]
        else:
            prompt_a, _ = obs_to_prompt(obs, agent="A")  # single agent always A
            action_a, _ = query_llm(prompt_a, think=think)
            actions = action_a

        obs, rewards, terminated, truncated, info = env.step(actions)
        frames.append(RolloutFrame(
            step=step, obs=obs, actions=actions, rewards=rewards,
            pixel_frame=get_pixel_frame(env, multiagent), info=info,
        ))
        print("time: ", time.time() - start_time)
        if terminated or truncated:
            break

    return frames