import time

import gym3
from torch.optim import Adam

import minipi.logger as logger
from minipi.algos.dqn.agent import DQNWorker
from minipi.algos.dqn.policy import DQNDiscretePolicy
from minipi.network.common import MLP
from minipi.buffer.prioritized import PrioritizedReplayBuffer
from minipi.utils.schedulers import LinearScheduler
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
    worker = DQNWorker(
        env_fn=make_gym3_env,
        env_kwargs={
            "num": 1,
            "env_kwargs": {"id": "CartPole-v0"},
            "use_subproc": False,
        },
        policy_fn=DQNDiscretePolicy,
        policy_kwargs={
            "extractor_fn": MLP,
            "extractor_kwargs": {"input_dim": 4, "hiddens": (128, 128)},
            "n_actions": 2,
            "dueling": True,
            "double_q": True,
        },
        optimizer_fn=Adam,
        optimizer_kwargs={"lr": 1e-3},
        n_step_return=1,
        exploration_eps={
            "scheduler_fn": LinearScheduler,
            "scheduler_kwargs": {
                "schedule_steps": 10_000,
                "value": 1.0,
                "final_value": 0.02,
            },
        },
        batch_size=32,
        discount_gamma=0.99,
        device="cuda:0",
    )

    # Create buffer
    buffer = PrioritizedReplayBuffer(capacity=50_000)

    # Training
    n_timesteps = 100_000
    train_freq = 1
    target_update_freq = 500
    learning_starts = 1000
    t_start = time.perf_counter()
    for i in range(n_timesteps):
        # Collect data
        worker.collect(i, buffer)
        # Learn on data
        if (i + 1) >= learning_starts:
            if (i + 1) % train_freq == 0:
                stats_dict = worker.learn(scheduler_step=i, buffer=buffer)
            if (i + 1) % target_update_freq == 0:
                worker.policy.update_target_net()
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
