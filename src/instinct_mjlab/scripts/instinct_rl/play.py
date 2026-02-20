"""Play Instinct-RL policies on migrated mjlab tasks.

Original: InstinctLab/scripts/instinct_rl/play.py
Migrated: replaces Isaac Sim / Isaac Lab runtime with mjlab + tyro CLI.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from instinct_rl.runners import OnPolicyRunner

import instinct_mjlab.tasks  # noqa: F401
import mjlab
from instinct_mjlab.rl import InstinctRlVecEnvWrapper
from instinct_mjlab.utils.distillation import prepare_distillation_algorithm_cfg
from instinct_mjlab.utils.motion_validation import (
  find_default_tracking_motion_file,
  resolve_datasets_root,
  validate_tracking_motion_file,
)
from instinct_mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_instinct_rl_cfg,
  load_runner_cls,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  motion_file: str | None = None
  registry_name: str | None = None
  checkpoint_file: str | None = None
  load_run: str | None = None
  checkpoint_pattern: str | None = None
  num_envs: int | None = None
  device: str | None = None
  viewer: Literal["auto", "native", "viser", "none"] = "auto"
  max_steps: int | None = None
  video: bool = False
  video_length: int = 400
  video_dir: str | None = None
  video_height: int | None = None
  video_width: int | None = None
  export_onnx: bool = False
  onnx_output_dir: str | None = None
  no_terminations: bool = False


class _ViewerEnvAdapter:
  """Adapter so mjlab viewers can consume instinct_rl vec envs."""

  def __init__(self, vec_env: InstinctRlVecEnvWrapper):
    self._vec_env = vec_env
    self.num_envs = vec_env.num_envs

  @property
  def device(self):
    return self._vec_env.device

  @property
  def cfg(self):
    return self._vec_env.cfg

  @property
  def unwrapped(self):
    return self._vec_env.unwrapped

  def get_observations(self):
    obs, _ = self._vec_env.get_observations()
    return obs

  def step(self, actions):
    return self._vec_env.step(actions)

  def reset(self):
    return self._vec_env.reset()

  def close(self):
    return self._vec_env.close()


def _resolve_device(device: str | None) -> str:
  if device is not None:
    return device
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_viewer_backend(viewer: str) -> str:
  if viewer != "auto":
    return viewer
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


def _resolve_rollout_steps(cfg: PlayConfig) -> int | None:
  if cfg.max_steps is not None:
    return cfg.max_steps
  if cfg.viewer == "none":
    return cfg.video_length if cfg.video else 300
  return None


def _resolve_tracking_motion(task_id: str, cfg: PlayConfig, env_cfg) -> None:
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )
  if not is_tracking_task:
    return

  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  if cfg.motion_file is not None:
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    validate_tracking_motion_file(motion_path)
    motion_cmd.motion_file = str(motion_path)
    return

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
    return

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
      return

  default_motion = find_default_tracking_motion_file(task_id)
  if default_motion is not None:
    motion_cmd.motion_file = str(default_motion)
    print(f"[INFO] Auto-selected motion file: {default_motion}")
    return

  raise ValueError(
    "Tracking 任务在回放时必须指定 motion：\n"
    "  --motion-file /path/to/motion.npz\n"
    "  或 --registry-name your-org/motions/motion-name\n"
    f"当前默认搜索目录: {resolve_datasets_root()} （可用 INSTINCT_DATASETS_ROOT 覆盖）"
  )


def _build_dummy_policy(agent_mode: str, action_shape: tuple[int, ...], device: str):
  if agent_mode == "zero":

    def zero_policy(_obs: torch.Tensor) -> torch.Tensor:
      return torch.zeros(action_shape, device=device)

    return zero_policy

  def random_policy(_obs: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.rand(action_shape, device=device) - 1.0

  return random_policy


def _resolve_checkpoint(
  task_id: str,
  cfg: PlayConfig,
  agent_cfg,
) -> Path:
  if cfg.checkpoint_file is not None:
    checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint.exists():
      raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint

  log_root_path = Path("logs") / "instinct_rl" / agent_cfg.experiment_name
  run_regex = cfg.load_run if cfg.load_run is not None else agent_cfg.load_run
  checkpoint_regex = (
    cfg.checkpoint_pattern
    if cfg.checkpoint_pattern is not None
    else agent_cfg.load_checkpoint
  )
  checkpoint = get_checkpoint_path(
    log_path=log_root_path,
    run_dir=run_regex,
    checkpoint=checkpoint_regex,
  )
  print(f"[INFO] Auto-selected checkpoint for {task_id}: {checkpoint}")
  return checkpoint


def _resolve_video_dir(
  *,
  cfg: PlayConfig,
  task_id: str,
  agent_cfg,
  checkpoint: Path | None,
) -> Path:
  if cfg.video_dir is not None:
    return Path(cfg.video_dir).expanduser().resolve()
  if checkpoint is not None:
    return checkpoint.parent / "videos" / "play"
  return Path("logs") / "instinct_rl" / agent_cfg.experiment_name / task_id / "videos" / "play"


def _resolve_onnx_output_dir(
  *,
  cfg: PlayConfig,
  task_id: str,
  agent_cfg,
  checkpoint: Path | None,
) -> Path:
  if cfg.onnx_output_dir is not None:
    return Path(cfg.onnx_output_dir).expanduser().resolve()
  if checkpoint is not None:
    return checkpoint.parent / "exported"
  return Path("logs") / "instinct_rl" / agent_cfg.experiment_name / task_id / "exported"


def _export_policy_to_onnx(
  *,
  runner,
  vec_env: InstinctRlVecEnvWrapper,
  task_id: str,
  export_dir: Path,
) -> None:
  export_dir.mkdir(parents=True, exist_ok=True)
  observations, _ = vec_env.get_observations()
  try:
    runner.export_as_onnx(obs=observations, export_model_dir=str(export_dir))
  except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
      "ONNX 导出失败：缺少依赖。请安装 `onnx` 后重试。"
    ) from err

  metadata = {
    "task_id": task_id,
    "obs_format": vec_env.get_obs_format(),
    "num_actions": vec_env.num_actions,
    "num_rewards": vec_env.num_rewards,
  }
  metadata_path = export_dir / "metadata.json"
  metadata_path.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  print(f"[INFO] Exported ONNX artifacts to: {export_dir}")


def _run_headless_rollout(
  env: _ViewerEnvAdapter,
  policy,
  *,
  num_steps: int | None,
) -> None:
  steps = num_steps if num_steps is not None else 300
  observations = env.get_observations()
  for _ in range(steps):
    with torch.no_grad():
      actions = policy(observations)
      observations, _, _, _ = env.step(actions)


def run_play(task_id: str, cfg: PlayConfig) -> None:
  if InstinctRlVecEnvWrapper is None:
    raise ImportError(
      "InstinctRlVecEnvWrapper is unavailable. Please install runtime deps:\n"
      "  pip install -e ../mjlab\n"
      "  pip install -e ../instinct_rl"
    )
  configure_torch_backends()
  viewer_backend = _resolve_viewer_backend(cfg.viewer)
  # Native viewer should use glfw; headless/video paths should use egl.
  if viewer_backend == "native":
    os.environ["MUJOCO_GL"] = "glfw"
  else:
    os.environ.setdefault("MUJOCO_GL", "egl")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_instinct_rl_cfg(task_id)
  device = _resolve_device(cfg.device)
  checkpoint: Path | None = None

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO] All terminations are disabled for play.")

  _resolve_tracking_motion(task_id, cfg, env_cfg)
  if cfg.agent == "trained":
    checkpoint = _resolve_checkpoint(task_id, cfg, agent_cfg)
  elif cfg.export_onnx:
    raise ValueError("`--export-onnx` 仅支持 `--agent trained`。")

  env = ManagerBasedRlEnv(
    cfg=env_cfg,
    device=device,
    render_mode="rgb_array" if cfg.video else None,
  )
  if cfg.video:
    video_dir = _resolve_video_dir(
      cfg=cfg,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint=checkpoint,
    )
    env = VideoRecorder(
      env,
      video_folder=video_dir,
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
      name_prefix=task_id.replace("/", "_"),
    )
    print(f"[INFO] Recording play video to: {video_dir}")

  vec_env = InstinctRlVecEnvWrapper(
    env,
    policy_group=agent_cfg.policy_observation_group,
    critic_group=agent_cfg.critic_observation_group,
  )

  viewer_env = _ViewerEnvAdapter(vec_env)

  if cfg.agent in {"zero", "random"}:
    policy = _build_dummy_policy(cfg.agent, (vec_env.num_envs, vec_env.num_actions), device)
  else:
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    agent_cfg_dict = asdict(agent_cfg)
    prepare_distillation_algorithm_cfg(
      agent_cfg=agent_cfg_dict,
      obs_format=vec_env.get_obs_format(),
      num_actions=vec_env.num_actions,
      num_rewards=vec_env.num_rewards,
    )
    runner = runner_cls(
      vec_env,
      agent_cfg_dict,
      log_dir=None,
      device=device,
    )
    assert checkpoint is not None
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] Loaded checkpoint: {checkpoint}")
    if cfg.export_onnx:
      onnx_output_dir = _resolve_onnx_output_dir(
        cfg=cfg,
        task_id=task_id,
        agent_cfg=agent_cfg,
        checkpoint=checkpoint,
      )
      _export_policy_to_onnx(
        runner=runner,
        vec_env=vec_env,
        task_id=task_id,
        export_dir=onnx_output_dir,
      )

  rollout_steps = _resolve_rollout_steps(cfg)

  if viewer_backend == "native":
    NativeMujocoViewer(viewer_env, policy).run(num_steps=rollout_steps)
  elif viewer_backend == "viser":
    ViserPlayViewer(viewer_env, policy).run(num_steps=rollout_steps)
  elif viewer_backend == "none":
    _run_headless_rollout(
      viewer_env,
      policy,
      num_steps=rollout_steps,
    )
  else:
    raise RuntimeError(f"Unsupported viewer backend: {viewer_backend}")

  viewer_env.close()


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
