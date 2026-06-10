from gymnasium.spaces import Discrete, Box
from numpy import inf, uint8

from gymnasium_stag_hunt.envs.gym.abstract_markov_staghunt import AbstractMarkovStagHuntEnv
from gymnasium_stag_hunt.src.entities import TILE_SIZE
from gymnasium_stag_hunt.src.games.escalation_game import Escalation


class EscalationEnv(AbstractMarkovStagHuntEnv):
    def __init__(
        self,
        grid_size=(5, 5),
        screen_size=(600, 600),
        obs_type="image",
        enable_multiagent=True,
        opponent_policy="pursuit",
        load_renderer=False,
        streak_break_punishment_factor=0.5,
        max_timesteps=1000,
        flip_obs=True
    ):
        """
        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param screen_size: A (W, H) tuple corresponding to the pixel dimensions of the game window
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        :param max_timesteps: The total number of timesteps per episode (default 1000)
        :param flip_obs: Whether to invert agent positions when returning the second obs (default True)
        """
        total_cells = grid_size[0] * grid_size[1]
        if total_cells < 3:
            raise AttributeError(
                "Grid is too small. Please specify a larger grid size."
            )

        super(EscalationEnv, self).__init__(
            grid_size=grid_size, obs_type=obs_type, enable_multiagent=enable_multiagent
        )

        # Rendering and State Variables
        self.game_title = "escalation"
        self.streak_break_punishment_factor = streak_break_punishment_factor
        window_title = (
            "Gymnasium - Escalation (%d x %d)" % grid_size
        )  # create game representation
        self.game = Escalation(
            window_title=window_title,
            grid_size=grid_size,
            screen_size=screen_size,
            obs_type=obs_type,
            enable_multiagent=enable_multiagent,
            load_renderer=load_renderer,
            streak_break_punishment_factor=streak_break_punishment_factor,
            opponent_policy=opponent_policy,
            max_timesteps=max_timesteps,
            flip_obs=flip_obs
        )

        # Environment Config
        self.action_space = Discrete(5)  # up, down, left, right or stand
        if obs_type == "image":  # Observation is the rgb pixel array
            self.observation_space = Box(
                0,
                255,
                shape=(2, grid_size[0] * TILE_SIZE, grid_size[1] * TILE_SIZE, 3),
                dtype=uint8,
            )
        elif obs_type == "coords":
            self.observation_space = Box(0, max(grid_size), shape=(2, 6,), dtype=uint8)

        self.reward_range = (
            -inf,
            inf,
        )  # There is technically no limit on how high or low the reinforcement can be
