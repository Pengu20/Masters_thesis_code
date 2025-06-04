from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from .airl_FAKE import AIRL

ALGOS = {
    'gail': GAIL,
    'airl': AIRL
}
