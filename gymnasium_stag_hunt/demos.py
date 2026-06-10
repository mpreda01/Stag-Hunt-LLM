from time import sleep

from gymnasium_stag_hunt.envs.gym.escalation import EscalationEnv
from gymnasium_stag_hunt.envs.gym.harvest import HarvestEnv
from gymnasium_stag_hunt.envs.gym.hunt import HuntEnv
from gymnasium_stag_hunt.envs.gym.simple import SimpleEnv
from gymnasium_stag_hunt.src.games.abstract_grid_game import UP, LEFT, DOWN, RIGHT, STAND

ENVS = {
    "CLASSIC": SimpleEnv,
    "HUNT": HuntEnv,
    "HARVEST": HarvestEnv,
    "ESCALATION": EscalationEnv,
}


def print_ep(obs, reward, done, info):
    print({"observation": obs, "reward": reward, "simulation over": done, "info": info})


def dir_parse(key):
    d = {LEFT: "LEFT", UP: "UP", DOWN: "DOWN", RIGHT: "RIGHT", STAND: "STAND"}
    return d[key]


def manual_input():
    i = input()
    if i in ["w", "W"]:
        i = UP
    elif i in ["a", "A"]:
        i = LEFT
    elif i in ["s", "S"]:
        i = DOWN
    elif i in ["d", "D"]:
        i = RIGHT
    elif i in ["x", "X"]:
        i = STAND

    return i


ENV = "HUNT"
enable_multi = True

if __name__ == "__main__":
    env = ENVS[ENV](obs_type="image", enable_multiagent=enable_multi)
    obs = env.reset()
    for i in range(10000):
        actions = [env.action_space.sample(), env.action_space.sample()] if enable_multi else env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(actions=actions)
        print(rewards)
        # print_ep(obs, rewards, terminated, truncated, info)
        sleep(0.4)
        if ENV == "CLASSIC":
            env.render()
        else:
            env.render(mode="human")
        if terminated or truncated:
            obs = env.reset()
    env.close()
    quit()
