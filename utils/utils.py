from pathlib import Path
import cv2

def get_frame_from_renderer(env, multiagent=True):
	"""Get pixel frame from renderer directly."""
	if env.game.RENDERER:
		img_frame = env.game.RENDERER._update_render(return_observation=True)
		if multiagent and len(img_frame.shape) == 4:
			img_frame = img_frame[0]  # Use first agent's frame
		return img_frame
	return None

def save_rollout_video(policy_fn, env_factory, steps=1000, output_path="rollout.mp4", fps=2, agent_obs_type="coords", **env_kwargs):
	"""Run one rollout and save the rendered frames as a video.

	policy_fn(obs, env, info) -> action or [action, action]
	env_factory(**kwargs) -> env instance
	agent_obs_type: observation type for the policy (e.g., "coords", "image")
	               video is rendered independently from agent observations
	"""

	# Ensure renderer is loaded even if using coord observations
	env = env_factory(obs_type=agent_obs_type, load_renderer=True, **env_kwargs)
	obs, info = env.reset()

	multiagent = bool(getattr(env, "enable_multiagent", getattr(env, "_enable_multiagent", True)))
	
	# Get initial frame from renderer (independent of agent obs_type)
	img_frame = get_frame_from_renderer(env, multiagent)

	frame_height, frame_width = img_frame.shape[:2]
	output_path = Path(output_path)
	writer = cv2.VideoWriter(
		str(output_path),
		cv2.VideoWriter_fourcc(*"mp4v"),
		fps,
		(frame_width, frame_height),
	)

	try:
		writer.write(cv2.cvtColor(img_frame, cv2.COLOR_RGB2BGR))

		for _ in range(steps):
			actions = policy_fn(obs, env, info)
			obs, rewards, terminated, truncated, info = env.step(actions)
			
			# Get frame from renderer (independent of agent observation type)
			img_frame = get_frame_from_renderer(env, multiagent)
			
			writer.write(cv2.cvtColor(img_frame, cv2.COLOR_RGB2BGR))
			if terminated or truncated:
				break
	finally:
		writer.release()
		env.close()

	return output_path

