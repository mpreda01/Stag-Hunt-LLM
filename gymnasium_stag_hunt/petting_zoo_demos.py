from time import sleep

from gymnasium_stag_hunt.envs import (
    ZooHuntEnvironment,
    ZooHarvestEnvironment,
    ZooEscalationEnvironment,
)

ENVS = {
    "HUNT": ZooHuntEnvironment,
    "HARVEST": ZooHarvestEnvironment,
    "ESCALATION": ZooEscalationEnvironment,
}

ENV = "HARVEST"

if __name__ == "__main__":
    env = ENVS[ENV](obs_type="image", enable_multiagent=True)
    obs = env.reset()
    for i in range(100):
        actions = {agent: env._action_spaces[agent].sample() for agent in env.agents}
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(rewards)
        env.render()
        sleep(0.4)
    env.close()
    quit()
