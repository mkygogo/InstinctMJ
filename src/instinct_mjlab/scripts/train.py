"""Train Instinct-RL policies on top of mjlab environments."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro
from instinct_rl.runners import OnPolicyRunner

import instinct_mjlab.tasks  # noqa: F401
import mjlab
from instinct_mjlab.rl import InstinctRlOnPolicyRunnerCfg, InstinctRlVecEnvWrapper
from instinct_mjlab.scripts._utils import (
  find_default_tracking_motion_file,
  prepare_distillation_algorithm_cfg,
  resolve_datasets_root,
  to_serializable,
  validate_tracking_motion_file,
  validate_distillation_runtime_cfg,
  validate_distillation_teacher_assets,
)
from instinct_mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_instinct_rl_cfg,
  load_runner_cls,
)
from instinct_mjlab.envs import InstinctRlEnv
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: InstinctRlOnPolicyRunnerCfg
  motion_file: str | None = None
  registry_name: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2_000
  gpu_ids: list[int] | Literal["all"] | None = None

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    use_play_cfg = task_id.endswith("-Play-v0")
    return TrainConfig(
      env=load_env_cfg(task_id, play=use_play_cfg),
      agent=load_instinct_rl_cfg(task_id),
    )


def _resolve_tracking_motion(task_id: str, cfg: TrainConfig) -> str | None:
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )
  if not is_tracking_task:
    return None

  motion_cmd = cfg.env.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  if cfg.motion_file is not None:
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    validate_tracking_motion_file(motion_path)
    motion_cmd.motion_file = str(motion_path)
    return None

  if cfg.registry_name:
    registry_name = cfg.registry_name
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    motion_path = (Path(artifact.download()) / "motion.npz").resolve()
    validate_tracking_motion_file(motion_path)
    motion_cmd.motion_file = str(motion_path)
    return registry_name

  configured_motion = str(getattr(motion_cmd, "motion_file", "")).strip()
  if configured_motion:
    configured_path = Path(configured_motion).expanduser().resolve()
    try:
      validate_tracking_motion_file(configured_path)
    except (ValueError, OSError, FileNotFoundError):
      pass
    else:
      motion_cmd.motion_file = str(configured_path)
      print(f"[INFO] Using motion file from env config: {configured_path}")
      return None

  default_motion = find_default_tracking_motion_file(task_id)
  if default_motion is not None:
    motion_cmd.motion_file = str(default_motion)
    print(f"[INFO] Auto-selected motion file: {default_motion}")
    return None

  raise ValueError(
    "Tracking 任务必须提供 motion 数据：\n"
    "  --motion-file /path/to/motion.npz\n"
    "  或 --registry-name your-org/motions/motion-name\n"
    f"当前默认搜索目录: {resolve_datasets_root()} （可用 INSTINCT_DATASETS_ROOT 覆盖）"
  )


def _resolve_device(cfg: TrainConfig) -> str:
  if cfg.device is not None:
    return cfg.device
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  if InstinctRlVecEnvWrapper is None:
    raise ImportError(
      "InstinctRlVecEnvWrapper is unavailable. Please install runtime deps:\n"
      "  pip install -e ../mjlab\n"
      "  pip install -e ../instinct_rl"
    )
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()

  device = _resolve_device(cfg)
  if device.startswith("cpu"):
    raise ValueError(
      "instinct_rl 当前训练链路依赖 CUDA 统计接口，请使用 GPU 设备，例如 --device cuda:0。"
    )
  cfg.agent.device = device
  cfg.env.seed = cfg.agent.seed
  if cfg.num_envs is not None:
    cfg.env.scene.num_envs = cfg.num_envs

  registry_name = _resolve_tracking_motion(task_id, cfg)

  print(f"[INFO] Task={task_id}, device={device}, num_envs={cfg.env.scene.num_envs}")
  print(f"[INFO] Logging to: {log_dir}")

  env = InstinctRlEnv(
    cfg=cfg.env,
    device=device,
    render_mode="rgb_array" if cfg.video else None,
  )

  if cfg.video:
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  vec_env = InstinctRlVecEnvWrapper(
    env,
    policy_group=cfg.agent.policy_observation_group,
    critic_group=cfg.agent.critic_observation_group,
  )

  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  agent_cfg_dict = asdict(cfg.agent)
  obs_format = vec_env.get_obs_format()
  prepare_distillation_algorithm_cfg(
    agent_cfg=agent_cfg_dict,
    obs_format=obs_format,
    num_actions=vec_env.num_actions,
    num_rewards=vec_env.num_rewards,
  )
  validate_distillation_runtime_cfg(
    agent_cfg=agent_cfg_dict,
    obs_format=obs_format,
    num_actions=vec_env.num_actions,
    num_rewards=vec_env.num_rewards,
  )
  teacher_checkpoint = validate_distillation_teacher_assets(agent_cfg=agent_cfg_dict)
  if teacher_checkpoint is not None:
    print(f"[INFO] Using teacher checkpoint: {teacher_checkpoint}")

  runner = runner_cls(
    vec_env,
    agent_cfg_dict,
    log_dir=str(log_dir),
    device=cfg.agent.device,
  )
  if hasattr(runner, "add_git_repo_to_log"):
    runner.add_git_repo_to_log(__file__)

  if cfg.agent.resume:
    log_root_path = log_dir.parent
    load_run = Path(cfg.agent.load_run).expanduser() if cfg.agent.load_run else None
    if load_run is not None and load_run.is_absolute():
      resume_path = get_checkpoint_path(
        log_path=load_run.parent,
        run_dir=load_run.name,
        checkpoint=cfg.agent.load_checkpoint,
      )
    else:
      resume_path = get_checkpoint_path(
        log_path=log_root_path,
        run_dir=cfg.agent.load_run,
        checkpoint=cfg.agent.load_checkpoint,
      )
    print(f"[INFO] Resuming from checkpoint: {resume_path}")
    runner.load(str(resume_path))

  dump_yaml(log_dir / "params" / "env.yaml", to_serializable(cfg.env))
  dump_yaml(log_dir / "params" / "agent.yaml", to_serializable(cfg.agent))
  if registry_name is not None:
    dump_yaml(
      log_dir / "params" / "registry.yaml",
      {"registry_name": registry_name},
    )

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations,
    init_at_random_ep_len=True,
  )
  vec_env.close()


def launch_training(task_id: str, args: TrainConfig | None = None) -> None:
  args = args or TrainConfig.from_task(task_id)
  log_root_path = Path("logs") / "instinct_rl" / args.agent.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name
  run_train(task_id=task_id, cfg=args, log_dir=log_dir)


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
