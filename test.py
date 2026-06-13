from utils.utils import save_rollout_video, save_rollout_csv, run_random_rollout, obs_to_prompt
from utils.const import ENV_FACTORIES
from agents.random import random_policy
    
    
  


if __name__ == "__main__":
    
    MULTIAGENT = True
    frames = run_random_rollout(
        policy_fn=random_policy,
        env_factory=ENV_FACTORIES["hunt"],
        agent_obs_type="coords",
        steps=2,
        enable_multiagent=MULTIAGENT,
    )

    print(f"Collected {len(frames)} frames, shape: {frames[0].pixel_frame.shape}")
    
    save_rollout_csv(multiagent=MULTIAGENT, frames=frames, output_path="hunt_rollout.csv")
    print("Rollout csv saved")

    # Save BEFORE anything that might call env.close()
    video_path = save_rollout_video(frames=frames, output_path="hunt_rollout.mp4", fps=4)
    print(f"Video saved: {video_path}")

    print("observations:\n", frames[0].obs, "\n\n", frames[1].obs)
    promptA, promptB = obs_to_prompt(frames[0].obs, agent="A")
    print("Prompt:\n", promptA, "\n\n", promptB)