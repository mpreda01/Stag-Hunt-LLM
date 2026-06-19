from utils.const import ENV_FACTORIES




def sample_actions(env, multiagent=True):
	"""Sample one action or one action per agent from the env action space."""
	if multiagent:
		return [env.action_space.sample(), env.action_space.sample()]
	return env.action_space.sample()

def random_policy(obs, env, info):
	multiagent = bool(getattr(env, "enable_multiagent", getattr(env, "_enable_multiagent", True)))
	return sample_actions(env, multiagent=multiagent)


def run_random_episode(env_name="hunt", steps=1000, multiagent=True, **env_kwargs):
	"""Run a random policy in one of the grid environments."""
	key = env_name.lower()
	if key not in ENV_FACTORIES:
		raise ValueError(f"Unknown env_name: {env_name!r}. Expected one of {sorted(ENV_FACTORIES)}")

	env = ENV_FACTORIES[key](enable_multiagent=multiagent, **env_kwargs)
	_, _ = env.reset()

	history = []
	for _ in range(steps):
		actions = sample_actions(env, multiagent=multiagent)
		obs, rewards, terminated, truncated, info = env.step(actions)
		history.append((actions, rewards))
		if terminated or truncated:
			_, _ = env.reset()

	env.close()
	return history



