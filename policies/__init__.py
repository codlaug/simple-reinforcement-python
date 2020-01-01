from policies.actor_critic import ActorCriticPolicy
# from policies.ddpg import DDPGPolicy
from policies.dqn import DqnPolicy
from policies.ppo import PPOPolicy
# from policies.qlearning import QlearningPolicy
# from policies.reinforce import ReinforcePolicy

ALL_POLICIES = [
    ActorCriticPolicy,
    # DDPGPolicy,
    DqnPolicy,
    PPOPolicy,
    # QlearningPolicy,
    # ReinforcePolicy
]