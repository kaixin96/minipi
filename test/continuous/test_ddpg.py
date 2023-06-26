import time

import gym
import gym3
from torch.optim import Adam

import minipi.logger as logger
from minipi.algos.ddpg.agent import DDPGWorker
from minipi.algos.ddpg.policy import DDPGContinuousPolicy
from minipi.utils.noise import GaussianActionNoise
from minipi.network.common import MLP
from minipi.buffer.common import UniformReplayBuffer
from minipi.envs.gym_wrapper import ScaleActionWrapper
from minipi.envs.gym3_wrapper import EpisodeStatsWrapper


def make_gym_env(**env_kwargs):
    env = gym.make(**env_kwargs)
    env = ScaleActionWrapper(env)
    return env


def make_gym3_env(**kwargs):
    env = gym3.vectorize_gym(**kwargs)
    env = EpisodeStatsWrapper(env)
    return env


def test():
    # Set up logger
    logger.configure(format_strs=["stdout"])
    logger.set_level(logger.ERROR)

    # Create worker
    worker = DDPGWorker(
        env_fn=make_gym3_env,
        env_kwargs={
            "num": 1,
            "env_fn": make_gym_env,
            "env_kwargs": {"id": "Pendulum-v1"},
            "use_subproc": False,
        },
        policy_fn=DDPGContinuousPolicy,
        policy_kwargs={
            "extractor_fn": MLP,
            "extractor_kwargs": {"input_dim": 3, "hiddens": ()},
            "action_dim": 1,
            "actor_hiddens": (400, 300),
            "critic_hiddens": (400, 300),
            "target_update_ratio": 0.005,
        },
        action_noise_fn=GaussianActionNoise,
        warmup_steps=500,
        optimizer_fn=Adam,
        optimizer_kwargs={"actor": {"lr": 1e-4}, "critic": {"lr": 1e-3}},
        batch_size=128,
        discount_gamma=0.99,
        device="cuda:0",
    )

    # Create buffer
    buffer = UniformReplayBuffer(capacity=int(5e4))

    # Training
    n_timesteps = int(5e4)
    t_start = time.perf_counter()
    for i in range(n_timesteps):
        # Collect data
        worker.collect(i, buffer)
        # Learn on data
        if i > worker.warmup_steps:
            stats_dict = worker.learn(i, buffer)
            if (i + 1) % 1000 == 0:
                ret = worker.env.callmethod("get_ep_stat_mean", "r")
                logger.logkv("steps", i + 1)
                logger.logkv("time", time.perf_counter() - t_start)
                logger.logkv("return", ret)
                for key, value in stats_dict.items():
                    logger.logkv(key, value)
                logger.dumpkvs()
                t_start = time.perf_counter()


if __name__ == "__main__":
    test()
