from utils.utils import save_rollout_video, save_rollout_csv, run_llm_rollout, run_random_rollout
from utils.const import ENV_FACTORIES
    
    
  


if __name__ == "__main__":
    
    MULTIAGENT = True
    TASK = "hunt"
    N_EPISODES = 20
    
    mode = input("Select agent (press Enter to continue):\n1. Random (default)\n2. Qwen4b zero shot\n3. Qwen4b one shot\n 4. Qwen4b few shot\n> ")
    output_path = input("Enter output path for rollout: ")
    if not output_path:
        output_path = ""
    if mode == "2" or mode == "3" or mode == "4":
        print("Using Qwen4b agent")
        for i in range(N_EPISODES):
            frames = run_llm_rollout(
                    env_factory=ENV_FACTORIES["hunt"],
                    steps=200,
                    agent_obs_type="coords",
                    think=False,
                    prompot_type=mode,
                ) 
            
            print(f"Collected {len(frames)} frames, shape: {frames[0].pixel_frame.shape}")

            save_rollout_csv(multiagent=MULTIAGENT, frames=frames, output_path=output_path + f"hunt_rollout_{i}.csv")
            print("Rollout csv saved")

            # Save BEFORE anything that might call env.close()
            video_path = save_rollout_video(frames=frames, output_path=output_path + f"hunt_rollout_{i}.mp4", fps=4)
            print(f"Video saved: {video_path}")
            
    else:
        print("Using random agent")
        for i in range(N_EPISODES):
         
            frames = run_random_rollout(
                env_factory=ENV_FACTORIES[TASK],
                steps=200,
                agent_obs_type="coords",
            )

            print(f"Collected {len(frames)} frames, shape: {frames[0].pixel_frame.shape}")
            
            save_rollout_csv(multiagent=MULTIAGENT, frames=frames, output_path=output_path + f"hunt_rollout_{i}.csv")
            print("Rollout csv saved")

            # Save BEFORE anything that might call env.close()
            video_path = save_rollout_video(frames=frames, output_path=output_path + f"hunt_rollout_{i}.mp4", fps=4)
            print(f"Video saved: {video_path}")


    