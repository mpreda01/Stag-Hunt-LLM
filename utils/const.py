from gymnasium_stag_hunt.envs.gym.escalation import EscalationEnv
from gymnasium_stag_hunt.envs.gym.harvest import HarvestEnv
from gymnasium_stag_hunt.envs.gym.hunt import HuntEnv


ENV_FACTORIES = {
	"hunt": HuntEnv,
	"escalation": EscalationEnv,
	"harvest": HarvestEnv,
}

SYSTEM_PROMPT = """You are an agent playing a cooperative grid-based Hunt game.

GOAL: Coordinate with your teammate to catch the Stag by both occupying its cell at the same time.
- If BOTH agents step on the Stag together → both get +5 reward (success)
- If YOU step on the Stag alone → you get -5 punishment (avoid this)
- If you step on a Plant → you get +1 reward (safe but suboptimal)

The grid is 5x5. Coordinates are (x, y) where (0,0) is top-left, x increases rightward, y increases downward.

Always reason step by step, then output your action on the last line in the format:
ACTION: <LEFT|DOWN|RIGHT|UP>

"""

ACTION_MAP = {
    "LEFT": 0,
    "DOWN": 1,
    "RIGHT": 2,
    "UP": 3,
}