import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

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