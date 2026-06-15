from utils.utils import save_rollout_video, save_rollout_csv, run_llm_rollout, obs_to_prompt
from utils.const import ENV_FACTORIES
from agents.qwen4b import query_llm
    
    
  


if __name__ == "__main__":
    
    MULTIAGENT = True
    frames = run_llm_rollout(
                env_factory=ENV_FACTORIES["hunt"],
                steps=10,
                agent_obs_type="coords",
                think=False,
            ) 

    print(f"Collected {len(frames)} frames, shape: {frames[0].pixel_frame.shape}")
    
    save_rollout_csv(multiagent=MULTIAGENT, frames=frames, output_path="hunt_rollout.csv")
    print("Rollout csv saved")

    # Save BEFORE anything that might call env.close()
    video_path = save_rollout_video(frames=frames, output_path="hunt_rollout.mp4", fps=4)
    print(f"Video saved: {video_path}")


    