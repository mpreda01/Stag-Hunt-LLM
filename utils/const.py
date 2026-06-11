from gymnasium_stag_hunt.envs.gym.escalation import EscalationEnv
from gymnasium_stag_hunt.envs.gym.harvest import HarvestEnv
from gymnasium_stag_hunt.envs.gym.hunt import HuntEnv


ENV_FACTORIES = {
	"hunt": HuntEnv,
	"escalation": EscalationEnv,
	"harvest": HarvestEnv,
}