from utils.utils import RolloutFrame, get_pixel_frame, save_rollout_video, save_rollout_csv
from utils.const import ENV_FACTORIES
from agents.random import random_policy
    
    
def run_rollout(policy_fn, env_factory, steps=200, agent_obs_type="coords", **env_kwargs) -> list[RolloutFrame]:
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


if __name__ == "__main__":
    
    MULTIAGENT = True
    frames = run_rollout(
        policy_fn=random_policy,
        env_factory=ENV_FACTORIES["hunt"],
        agent_obs_type="image",
        steps=200,
        enable_multiagent=MULTIAGENT,
    )

    print(f"Collected {len(frames)} frames, shape: {frames[0].pixel_frame.shape}")
    
    save_rollout_csv(multiagent=MULTIAGENT, frames=frames, output_path="hunt_rollout.csv")
    print("Rollout csv saved")

    # Save BEFORE anything that might call env.close()
    video_path = save_rollout_video(frames=frames, output_path="hunt_rollout.mp4", fps=4)
    print(f"Video saved: {video_path}")

