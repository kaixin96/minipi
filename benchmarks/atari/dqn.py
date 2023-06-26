"""
Follow hyperparameters in Dopamine except sticky actions

https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn.gin
"""

import os
import json
import time
from functools import partial

import numpy as np
import torch as th
from torch.optim import RMSprop

import minipi.logger as logger
from minipi.algos.dqn.agent import DQNWorker
from minipi.algos.dqn.policy import DQNDiscretePolicy
from minipi.buffer.common import UniformReplayBuffer
from minipi.envs.prebuilt_envs import atari_vec_env
from minipi.envs.gym3_wrapper import (
    ObsTransposeWrapper,
    CollectEpisodeStatsWrapper,
    FrameStackWrapper,
)
from minipi.network.common import NatureCNN
from minipi.network.initialization import lasagne_orthogonal_init as orthogonal_init
from minipi.utils.schedulers import LinearScheduler


def make_atari_env(**kwargs):
    env = atari_vec_env(**kwargs)
    env = FrameStackWrapper(env, n_stack=4, stack_axis=-1)
    env = ObsTransposeWrapper(env, axes=(2, 0, 1))
    env = CollectEpisodeStatsWrapper(env, ep_stat_keys=("r",))
    return env


def run_dqn_worker(cfg):
    rank = cfg["comm_cfg"]["rank"]

    # Configure logger
    run_dir = os.path.join(cfg["run_cfg"]["log_dir"], f"run_{cfg['run_cfg']['run_id']}")
    os.makedirs(run_dir, exist_ok=True)
    logger.configure(run_dir, format_strs=["csv", "stdout"], log_suffix=f"-rank{rank}")
    # Save configurations
    if rank == 0:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=4, default=str)

    th.backends.cudnn.benchmark = True
    if cfg["comm_cfg"]["distributed"]:
        os.environ["MASTER_ADDR"] = cfg["comm_cfg"]["master_address"]
        os.environ["MASTER_PORT"] = cfg["comm_cfg"]["master_port"]
        th.distributed.init_process_group(
            cfg["comm_cfg"]["backend"], world_size=cfg["comm_cfg"]["world_size"], rank=rank
        )
    if cfg["eval_cfg"]["rank"] == rank:
        worker_kwargs = {**cfg["worker_kwargs"], **cfg["eval_cfg"]["eval_worker_kwargs"]}
    else:
        worker_kwargs = cfg["worker_kwargs"]
    
    # Initialize worker
    worker = DQNWorker(**worker_kwargs)

    # Create buffer
    buffer = cfg['buffer_cfg']['buffer_fn'](**cfg['buffer_cfg']['buffer_kwargs'])

    # Training
    for i in range(cfg["run_cfg"]["n_timesteps"]):
        tstart = time.perf_counter()
        # Collect data
        worker.collect(scheduler_step=i, buffer=buffer)
        # Learning
        if i >= cfg["run_cfg"]["learning_starts"]:
            if (i + 1) % cfg["run_cfg"]["train_freq"] == 0:
                stats_dict = worker.learn(scheduler_step=i, buffer=buffer)
            if (i + 1) % cfg["run_cfg"]["target_update_freq"] == 0:
                worker.policy.update_target_net()
            # Logging
            if (i + 1) % cfg["run_cfg"]["logging_freq"] == 0:
                logger.logkv("steps", i + 1)
                ret = worker.env.callmethod("get_ep_stat_mean", "r")
                for key, value in stats_dict.items():
                    logger.logkv(key, value)
                logger.logkv("return", ret)
                logger.logkv("time", time.perf_counter() - tstart)
                logger.dumpkvs()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--master_address", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="29500")
    parser.add_argument("--comm_backend", type=str, default="nccl", choices=["gloo", "nccl"])

    # env config
    parser.add_argument("--env_name", type=str, default="BreakoutNoFrameskip-v4")

    args = parser.parse_args()

    config = {
        # Communication for distributed computing
        "comm_cfg": {
            "distributed": False,
            "master_address": args.master_address,
            "master_port": args.master_port,
            "rank": args.rank,
            "world_size": args.world_size,
            "backend": args.comm_backend,
        },
        # Run
        "run_cfg": {
            "log_dir": f"./exps/atari/{args.env_name}",
            "run_id": args.run_id,
            "n_timesteps": 50_000_000,
            "learning_starts": 20000,
            "target_update_freq": 8000,
            "train_freq": 4,
            "logging_freq": 5000,
        },
        # Buffer
        "buffer_cfg": {
            "buffer_fn": UniformReplayBuffer,
            "buffer_kwargs": {"capacity": 1_000_000},
        },
        # Agent
        "worker_kwargs": {
            "env_fn": make_atari_env,
            "env_kwargs": {
                "num_envs": 1,
                "env_name": args.env_name,
                "use_subproc": False,
            },
            "policy_fn": DQNDiscretePolicy,
            "policy_kwargs": {
                "extractor_fn": NatureCNN,
                "extractor_kwargs": {"input_shape": (4, 84, 84)},
                "n_actions": 4,
                "init_weight_fn": lambda x: x.apply(partial(orthogonal_init, scale=np.sqrt(2))),
            },
            "optimizer_fn": RMSprop,
            "optimizer_kwargs": {
                "lr": 2.5e-4,
                "alpha": 0.95,
                "eps": 1e-5,
                "centered": True,
            },
            "n_step_return": 1,
            "exploration_eps": {
                "scheduler_fn": LinearScheduler,
                "scheduler_kwargs": {
                    "schedule_steps": 1_000_000,
                    "value": 1.0,
                    "final_value": 0.01,
                },
            },
            "batch_size": 32,
            "discount_gamma": 0.99,
            "worker_weight": 1.0,
            "device": f"cuda:{args.gpu_id}",
        },
        # Eval
        "eval_cfg": {
            "enabled": False,
            "rank": -1,
            "eval_worker_kwargs": {
                "exploration_eps": 0.001,
                "worker_weight": 0.0,
            },
        },
    }

    run_dqn_worker(config)

