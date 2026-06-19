from gymnasium_stag_hunt.envs.gym.escalation import EscalationEnv
from gymnasium_stag_hunt.envs.gym.harvest import HarvestEnv
from gymnasium_stag_hunt.envs.gym.hunt import HuntEnv


ENV_FACTORIES = {
	"hunt": HuntEnv,
	"escalation": EscalationEnv,
	"harvest": HarvestEnv,
}

ZERO_SHOT = """You are an agent playing a cooperative grid-based Hunt game.

GOAL: Coordinate with your teammate to catch the Stag by both occupying its cell at the same time.
- If BOTH agents step on the Stag together → both get +5 reward (success)
- If YOU step on the Stag alone → you get -5 punishment (avoid this)
- If you step on a Plant → you get +1 reward (safe but suboptimal, only if stag is unreachable)

ACTIONS: LEFT moves you (-1, 0), RIGHT moves you (+1, 0), UP moves you (0, -1), DOWN moves you (0, +1).
The grid is 5x5. Coordinates are (x, y) where (0,0) is top-left, x increases rightward, y increases downward.
You cannot move outside the grid boundaries.

Always reason step by step, then output your action on the last line in the format:
ACTION: <LEFT|DOWN|RIGHT|UP>

"""

ACTION_MAP = {
    "LEFT": 0,
    "DOWN": 1,
    "RIGHT": 2,
    "UP": 3,
}

ONE_SHOT = """You are an agent playing a cooperative grid-based Hunt game.

GOAL: Coordinate with your teammate to catch the Stag by both occupying its cell at the same time.
- If BOTH agents step on the Stag together → both get +5 reward (success)
- If YOU step on the Stag alone → you get -5 punishment (avoid this)
- If you step on a Plant → you get +1 reward (safe but suboptimal, only if stag is unreachable)

ACTIONS: LEFT moves you (-1, 0), RIGHT moves you (+1, 0), UP moves you (0, -1), DOWN moves you (0, +1).
The grid is 5x5. Coordinates are (x, y) where (0,0) is top-left, x increases rightward, y increases downward.
You cannot move outside the grid boundaries.

STRATEGY:
- Always prioritize catching the Stag over harvesting plants.
- The Stag moves toward the nearest agent each turn — use this to your advantage.
- Think about where your teammate is heading and converge on the Stag from opposite sides.
- If you and your teammate are both far from the Stag, move to cut off its path.

--- EXAMPLE ---

Situation:
- You (Agent A): (0, 0)
- Teammate (Agent B): (4, 0)
- Stag: (2, 2)
- Plants: (1, 4), (3, 3)

Reasoning:
The Stag is at (2,2), in the center. I am at (0,0) and my teammate B is at (4,0).
We are symmetric: I am on the left, B is on the right.
The Stag will move toward the nearest agent. Distance from Stag to me: |2-0|+|2-0|=4. Distance from Stag to B: |2-4|+|2-0|=4. Equal distance, so the Stag may move in any direction toward us.
The optimal plan: I move RIGHT to close in from the left, B moves LEFT to close in from the right. We converge on the Stag at (2,2) together.
Moving RIGHT takes me from (0,0) to (1,0), reducing my x-distance to the Stag from 2 to 1.
I should not go for the Plant at (1,4) — it gives only +1 and wastes turns.
Conclusion: move RIGHT to approach the Stag and set up a cooperative catch with Agent B.
ACTION: RIGHT

--- END EXAMPLE ---

Now reason step by step about your current situation, then output your action on the last line in the format:
ACTION: <LEFT|DOWN|RIGHT|UP>

"""

TWO_SHOT = """You are an agent playing a cooperative grid-based Hunt game.

GOAL: Coordinate with your teammate to catch the Stag by both occupying its cell at the same time.
- If BOTH agents step on the Stag together → both get +5 reward (success)
- If YOU step on the Stag alone → you get -5 punishment (avoid this)
- If you step on a Plant → you get +1 reward (safe but suboptimal, only if stag is unreachable)

ACTIONS: LEFT moves you (-1, 0), RIGHT moves you (+1, 0), UP moves you (0, -1), DOWN moves you (0, +1).
The grid is 5x5. Coordinates are (x, y) where (0,0) is top-left, x increases rightward, y increases downward.
You cannot move outside the grid boundaries.

STRATEGY:
- Always prioritize catching the Stag over harvesting plants.
- The Stag moves toward the nearest agent each turn — use this to your advantage.
- Think about where your teammate is heading and converge on the Stag from opposite sides.
- If you and your teammate are both far from the Stag, move to cut off its path.

--- EXAMPLE 1: Symmetric approach from corners ---

Situation:
- You (Agent A): (0, 0)
- Teammate (Agent B): (4, 0)
- Stag: (2, 2)
- Plants: (1, 4), (3, 3)

Reasoning:
I am at (0,0), my teammate B is at (4,0), and the Stag is at (2,2).
Manhattan distance from me to Stag: |2-0|+|2-0| = 4.
Manhattan distance from B to Stag: |2-4|+|2-0| = 4.
We are equidistant and symmetric around the Stag. The Stag will move toward the nearest agent — since we are equal, it may move in any direction.
The optimal strategy is to converge from opposite sides: I approach from the left, B approaches from the right. We aim to both land on (2,2) simultaneously.
Moving RIGHT takes me from (0,0) to (1,0), reducing my x-distance to the Stag by 1. This is the correct direction.
I should not chase the Plant at (1,4) — it costs turns and abandons the cooperative plan.
Conclusion: move RIGHT to approach the Stag and set up a cooperative catch with B.
ACTION: RIGHT

--- EXAMPLE 2: Midgame — stag between agents on the same row ---

Situation:
- You (Agent A): (0, 2)
- Teammate (Agent B): (4, 2)
- Stag: (2, 2)
- Plants: (1, 0), (3, 4)

Reasoning:
I am at (0,2), my teammate B is at (4,2), and the Stag is at (2,2).
All three are on the same row y=2. The Stag is exactly between us.
Manhattan distance from me to Stag: |2-0|+|2-2| = 2.
Manhattan distance from B to Stag: |2-4|+|2-2| = 2.
We are equidistant and the Stag is directly to my RIGHT and directly to B's LEFT.
The Stag will move toward the nearest agent. Since we are equidistant it may stay on row y=2 or shift slightly, but it cannot escape both of us if we both move inward.
My plan: move RIGHT from (0,2) to (1,2). Next turn, move RIGHT again to (2,2).
B should mirror: move LEFT from (4,2) to (3,2). Next turn, move LEFT again to (2,2).
We both arrive at (2,2) in exactly 2 steps — a clean cooperative catch.
I must not move DOWN or UP — that breaks the alignment and gives the Stag room to escape.
I must not go for the Plant at (1,0) — it is off-row and costs the cooperative timing.
Conclusion: move RIGHT to close in on the Stag along row y=2, converging with B from the opposite side.
ACTION: RIGHT

--- END EXAMPLES ---

Now reason step by step about your current situation, then output your action on the last line in the format:
ACTION: <LEFT|DOWN|RIGHT|UP>

"""