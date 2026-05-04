"""Runs UR3 block-pushing evaluation with a VersatIL policy client."""

import datetime
import logging
import os
from dataclasses import dataclass

import draccus
import wandb
from tso_robotics_sockets import ServerStatus, TransportKey

from versatil_inference.server import UR3BlockPushServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DATE_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class EvalConfig:
    """Configuration for UR3 block-pushing evaluation."""

    seed: int = 42
    num_trials: int = 50
    ip_address: str = "0.0.0.0"
    port: int = 5556
    compression_type: str = "raw"
    output_folder: str = ""
    max_parallel_envs: int = 10
    record_video: bool = False
    normalize_io: bool = False
    stats_path: str | None = None
    run_id_note: str | None = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "ur3-blockpush-eval"
    wandb_entity: str = ""


def run_evaluation(config: EvalConfig) -> None:
    run_id = f"EVAL-ur3-blockpush-{DATE_TIME}"
    if config.run_id_note:
        run_id += f"--{config.run_id_note}"
    os.makedirs(config.local_log_dir, exist_ok=True)
    if config.use_wandb:
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=run_id,
        )

    server = UR3BlockPushServer(
        ip_address=config.ip_address,
        port=config.port,
        compression_type=config.compression_type,
        seed=config.seed,
        num_trials=config.num_trials,
        output_folder=config.output_folder,
        max_parallel_envs=config.max_parallel_envs,
        record_video=config.record_video,
        normalize_io=config.normalize_io,
        stats_path=config.stats_path,
    )
    logging.info(
        f"UR3 eval: {config.num_trials} trials "
        f"(seeds starting at {config.seed}), "
        f"waiting for client on tcp://{config.ip_address}:{config.port}"
    )

    try:
        while True:
            response = server.handle_client_request()
            if response.get(TransportKey.STATUS.value) == ServerStatus.FINISHED.value:
                break
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        server.shutdown()

    rollout_directory = server.environment.rollout_directory
    rollout_directory.mkdir(parents=True, exist_ok=True)
    log_filepath = str(rollout_directory / "log.txt")
    _log_results(server=server, config=config, log_filepath=log_filepath)
    logging.info(f"Log saved to: {log_filepath}")


def _log_results(
    server: UR3BlockPushServer,
    config: EvalConfig,
    log_filepath: str,
) -> None:
    environment = server.environment
    if config.use_wandb:
        wandb.config.update(
            {
                "client_name": environment.client_name,
                "num_trials": config.num_trials,
                "seed": config.seed,
                "record_video": config.record_video,
                "normalize_io": config.normalize_io,
                "stats_path": config.stats_path,
            }
        )

    total_trials = sum(environment.number_of_resets)
    mean_reward = (
        sum(environment.environments_rewards) / total_trials
        if total_trials > 0
        else 0.0
    )
    mean_p1 = (
        sum(environment.environments_p1) / total_trials if total_trials > 0 else 0.0
    )
    mean_p2 = (
        sum(environment.environments_p2) / total_trials if total_trials > 0 else 0.0
    )
    mean_goals = mean_p1 + mean_p2
    behavior_entropy = environment.behavior_order_entropy()
    behavior_order_counts = environment.behavior_order_counts()

    with open(log_filepath, "w") as log_file:
        log_file.write(
            f"UR3 block-pushing evaluation - {config.num_trials} trials "
            f"(seeds {config.seed}..{config.seed + config.num_trials - 1})\n\n"
        )
        for i in range(environment.num_envs):
            episode_seed = environment._episode_seeds[i]
            reward = environment.environments_rewards[i]
            p1 = environment.environments_p1[i]
            p2 = environment.environments_p2[i]
            goals = p1 + p2
            behavior_order = environment.environments_behavior_order[i]
            log_file.write(
                f"Trial {i:3d} (seed={episode_seed}): "
                f"reward={reward:.4f}, "
                f"p1={int(p1)}, p2={int(p2)}, goals={int(goals)}, "
                f"behavior_order={behavior_order}\n"
            )
            if config.use_wandb:
                wandb.log(
                    {
                        "reward": reward,
                        "episode": i,
                        f"p1/trial_{i}": p1,
                        f"p2/trial_{i}": p2,
                        f"goals/trial_{i}": goals,
                    }
                )

        log_file.write(f"\nTrials: {total_trials}\n")
        log_file.write(f"Mean reward: {mean_reward:.4f}\n")
        log_file.write(f"Mean p1: {mean_p1:.4f}\n")
        log_file.write(f"Mean p2: {mean_p2:.4f}\n")
        log_file.write(f"Mean goals (/2): {mean_goals:.4f}\n")
        log_file.write(f"Behavior order entropy: {behavior_entropy:.4f}\n")
        log_file.write(f"Behavior order counts: {dict(behavior_order_counts)}\n")

    if config.use_wandb:
        wandb.log(
            {
                "mean_reward": mean_reward,
                "mean_p1": mean_p1,
                "mean_p2": mean_p2,
                "mean_goals": mean_goals,
                "behavior_order_entropy": behavior_entropy,
                "num_episodes/total": total_trials,
            }
        )
        for label, count in behavior_order_counts.items():
            wandb.log({f"behavior_order_count/{label}": count})

    logging.info(
        f"Mean reward: {mean_reward:.4f}, "
        f"mean p1: {mean_p1:.4f}, mean p2: {mean_p2:.4f}, "
        f"mean goals (/2): {mean_goals:.4f}, "
        f"behavior order entropy: {behavior_entropy:.4f} "
        f"over {total_trials} trials"
    )


@draccus.wrap()
def eval_ur3_blockpush(config: EvalConfig) -> None:
    run_evaluation(config=config)


if __name__ == "__main__":
    eval_ur3_blockpush()

