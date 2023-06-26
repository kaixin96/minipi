import json
import os
import time

import gym3
import torch as th
from procgen.env import ProcgenGym3Env, ENV_NAMES

import minipi.logger as logger
from minipi.algos.ppo.agent import PPOWorker
from minipi.algos.ppo.policy import PPODiscretePolicy
from minipi.buffer.common import Buffer
from minipi.network.impala import ImpalaCNN
from minipi.envs.gym3_wrapper import (
    ObsTransposeWrapper,
    EpisodeStatsWrapper,
    NormalizeWrapper,
    ClipWrapper,
)
from minipi.utils.preprocess import scale_preprocessor


def make_procgen_env(**kwargs):
    env = ProcgenGym3Env(**kwargs)
    env = gym3.ExtractDictObWrapper(env, "rgb")
    env = ObsTransposeWrapper(env, axes=(2, 0, 1))
    env = EpisodeStatsWrapper(env)
    env = NormalizeWrapper(env, normalize_obs=False)
    env = ClipWrapper(env, clip_rew=10.0)
    return env


def run_ppo_worker(cfg):
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
    worker = PPOWorker(**worker_kwargs)

    # Create buffer
    buffer = Buffer(capacity=worker.env.num, sequence_length=worker.n_steps)
    
    # Training
    for i in range(cfg["run_cfg"]["n_iters"]):
        tstart = time.perf_counter()
        # Collect data
        worker.collect(scheduler_step=i, buffer=buffer)
        collect_time = time.perf_counter() - tstart
        # Learn on data
        stats_dict = worker.learn(scheduler_step=i, buffer=buffer)
        learn_time = time.perf_counter() - tstart - collect_time
        # Logging
        if (i + 1) % cfg["run_cfg"]["logging_freq"] == 0:
            logger.logkv("iter", i + 1)
            for key, value in stats_dict.items():
                logger.logkv(key, value)
            ret = worker.env.callmethod("get_ep_stat_mean", "r")
            logger.logkv("return", ret)
            logger.logkv("time", time.perf_counter() - tstart)
            logger.logkv("collect_time", collect_time)
            logger.logkv("learn_time", learn_time)
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
    parser.add_argument("--env_name", type=str, choices=ENV_NAMES, required=True)
    parser.add_argument("--distribution_mode", type=str, default="easy", choices=["easy", "hard"])
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_levels", type=int, default=200)
    parser.add_argument("--start_level", type=int, default=0)
    parser.add_argument("--rand_seed", type=int, default=0)

    # ppo config
    parser.add_argument("--total_steps", type=int, default=25_000_000)
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--discount_gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--n_minibatches", type=int, default=8)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--vf_clip_range", type=float, default=0.2)
    parser.add_argument("--vf_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--logging_freq", type=int, default=1)
    args = parser.parse_args()

    cfg = {
        # Communication for distributed computing
        "comm_cfg": {
            "distributed": True,
            "master_address": args.master_address,
            "master_port": args.master_port,
            "rank": args.rank,
            "world_size": args.world_size,
            "backend": args.comm_backend,
        },
        # Run
        "run_cfg": {
            "log_dir": f"./exps/procgen/{args.env_name}",
            "run_id": args.run_id,
            "total_steps": args.total_steps,
            "n_iters": int(
                args.total_steps / args.num_envs / args.n_steps / (max(args.world_size - 1, 1))
            ),
            "logging_freq": args.logging_freq,
        },
        # Agent
        "worker_kwargs": {
            "env_fn": make_procgen_env,
            "env_kwargs": {
                "num": args.num_envs,
                "env_name": args.env_name,
                "distribution_mode": args.distribution_mode,
                "num_levels": args.num_levels,
                "start_level": args.start_level,
                "rand_seed": args.rand_seed,
            },
            "policy_fn": PPODiscretePolicy,
            "policy_kwargs": {
                "extractor_fn": ImpalaCNN,
                "extractor_kwargs": {"input_shape": (3, 64, 64), "depths": (16, 32, 32)},
                "n_actions": 15,
                "preprocess_obs_fn": scale_preprocessor,
                "preprocess_obs_kwargs": {"max_value": 255.0},
            },
            "n_steps": args.n_steps,
            "discount_gamma": args.discount_gamma,
            "gae_lambda": args.gae_lambda,
            "optimizer_fn": th.optim.Adam,
            "optimizer_kwargs": {"lr": args.lr, "eps": 1e-05},
            "n_epochs": args.n_epochs,
            "n_minibatches": args.n_minibatches,
            "normalize_adv": True,
            "clip_range": args.clip_range,
            "vf_clip_range": args.vf_clip_range,
            "vf_loss_coef": args.vf_loss_coef,
            "entropy_coef": args.entropy_coef,
            "max_grad_norm": args.max_grad_norm,
            "worker_weight": 1.0,
            "device": f"cuda:{args.gpu_id}",
        },
        # Eval
        "eval_cfg": {
            "rank": args.world_size - 1 if args.world_size > 1 else -1,
            "eval_worker_kwargs": {
                "env_kwargs": {
                    "num": args.num_envs,
                    "env_name": args.env_name,
                    "num_levels": 0,
                    "start_level": args.start_level + args.num_levels,
                    "rand_seed": args.rand_seed,
                },
                "worker_weight": 0.0,
            },
        },
    }

    run_ppo_worker(cfg=cfg)