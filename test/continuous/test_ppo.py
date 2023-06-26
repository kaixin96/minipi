import time

import gym
import gym3
from torch.optim import Adam

import minipi.logger as logger
from minipi.algos.ppo.agent import PPOWorker
from minipi.algos.ppo.policy import PPOContinuousPolicy
from minipi.network.common import MLP
from minipi.buffer.common import Buffer
from minipi.envs.gym_wrapper import ClipActionWrapper
from minipi.envs.gym3_wrapper import EpisodeStatsWrapper


def make_gym_env(**env_kwargs):
    env = gym.make(**env_kwargs)
    env = ClipActionWrapper(env)
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
    worker = PPOWorker(
        env_fn=make_gym3_env,
        env_kwargs={
            "num": 8,
            "env_fn": make_gym_env,
            "env_kwargs": {"id": "Pendulum-v1"},
            "use_subproc": False,
        },
        policy_fn=PPOContinuousPolicy,
        policy_kwargs={
            "extractor_fn": MLP,
            "extractor_kwargs": {"input_dim": 3, "hiddens": ()},
            "action_dim": 1,
            "actor_hiddens": (64, 64),
            "critic_hiddens": (64, 64),
        },
        optimizer_fn=Adam,
        optimizer_kwargs={"lr": 5e-4},
        n_steps=2048,
        n_epochs=10,
        n_minibatches=256,
        discount_gamma=0.99,
        gae_lambda=0.98,
        normalize_adv=True,
        clip_range=0.2,
        entropy_coef=0.0,
        device="cuda:0",
    )

    # Create buffer
    buffer = Buffer(capacity=worker.env.num, sequence_length=worker.n_steps)

    # Training
    n_iters = 100
    t_start = time.perf_counter()
    for i in range(n_iters):
        # Collect data
        worker.collect(scheduler_step=i, buffer=buffer)
        # Learn on data
        stats_dict = worker.learn(scheduler_step=i, buffer=buffer)
        # Logging
        ret = worker.env.callmethod("get_ep_stat_mean", "r")
        logger.logkv("iter", i + 1)
        logger.logkv("time", time.perf_counter() - t_start)
        logger.logkv("return", ret)
        for key, value in stats_dict.items():
            logger.logkv(key, value)
        logger.dumpkvs()
        t_start = time.perf_counter()


if __name__ == "__main__":
    test()
