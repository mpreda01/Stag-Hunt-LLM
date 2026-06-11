from utils.utils import save_rollout_video
from utils.const import ENV_FACTORIES
from agents.random import random_policy





if __name__ == "__main__":
	video_path = save_rollout_video(
		policy_fn=random_policy,
		env_factory=ENV_FACTORIES["hunt"],
		agent_obs_type="image",
		steps=200,
		output_path="hunt_rollout.mp4",
		enable_multiagent=False,
	)
