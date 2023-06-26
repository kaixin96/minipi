import time

import gym3
from torch.optim import Adam

import minipi.logger as logger
from minipi.algos.a2c.agent import A2CWorker
from minipi.algos.a2c.policy import A2CDiscretePolicy
from minipi.network.common import MLPRNN
from minipi.buffer.common import Buffer
from minipi.envs.gym3_wrapper import EpisodeStatsWrapper


def make_gym3_env(**kwargs):
    env = gym3.vectorize_gym(**kwargs)
    env = EpisodeStatsWrapper(env)
    return env


def test():
    # Set up logger
    logger.configure(format_strs=["stdout"])
    logger.set_level(logger.ERROR)

    # Create worker
    worker = A2CWorker(
        env_fn=make_gym3_env,
        env_kwargs={"num": 8, "env_kwargs": {"id": "CartPole-v0"}, "use_subproc": False,},
        policy_fn=A2CDiscretePolicy,
        policy_kwargs={
            "extractor_fn": MLPRNN,
            "extractor_kwargs": {
                "input_dim": 4,
                "hiddens": (128, 128),
                "rnn_hidden_size": 128,
                "num_rnn_layers": 2,
            },
            "n_actions": 2,
        },
        optimizer_fn=Adam,
        optimizer_kwargs={"lr": 5e-4},
        n_steps=5,
        discount_gamma=0.99,
        entropy_coef=0.0,
        device="cuda:0",
    )

    # Create buffer
    buffer = Buffer(capacity=worker.env.num, sequence_length=worker.n_steps)

    # Training
    n_iters = 10000
    t_start = time.perf_counter()
    for i in range(n_iters):
        # Collect data
        worker.collect(scheduler_step=i, buffer=buffer)
        # Learn on data
        stats_dict = worker.learn(scheduler_step=i, buffer=buffer)
        # Logging
        if (i + 1) % 100 == 0:
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
