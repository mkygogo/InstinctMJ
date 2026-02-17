"""Shared script helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from instinct_mjlab.utils.datasets import resolve_datasets_root as _resolve_datasets_root

_TRACKING_MOTION_REQUIRED_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def to_serializable(data: Any) -> Any:
  if is_dataclass(data):
    return {item.name: to_serializable(getattr(data, item.name)) for item in fields(data)}
  if isinstance(data, dict):
    return {str(key): to_serializable(value) for key, value in data.items()}
  if isinstance(data, tuple):
    return [to_serializable(value) for value in data]
  if isinstance(data, list):
    return [to_serializable(value) for value in data]
  if isinstance(data, (str, int, float, bool)) or data is None:
    return data
  return repr(data)


def resolve_datasets_root() -> Path:
  """Resolve datasets root for script entry points."""
  return _resolve_datasets_root()


def _tracking_motion_missing_keys(motion_path: Path) -> tuple[str, ...]:
  with np.load(motion_path) as data:
    keys = set(data.files)
  missing = tuple(key for key in _TRACKING_MOTION_REQUIRED_KEYS if key not in keys)
  return missing


def validate_tracking_motion_file(motion_path: Path) -> None:
  """Validate motion npz schema expected by `mjlab.tasks.tracking.mdp.MotionLoader`."""
  if not motion_path.exists() or not motion_path.is_file():
    raise FileNotFoundError(f"Motion file not found: {motion_path}")
  missing_keys = _tracking_motion_missing_keys(motion_path)
  if missing_keys:
    raise ValueError(
      "motion 文件格式不匹配 mjlab tracking 期望字段："
      f" {motion_path}\n"
      f"缺失 keys: {list(missing_keys)}\n"
      f"期望 keys: {list(_TRACKING_MOTION_REQUIRED_KEYS)}"
    )


def _find_parkour_motion_from_task_cfg() -> Path | None:
  """Resolve Parkour default motion from task config (InstinctLab-style path/yaml)."""
  try:
    from instinct_mjlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import (
      AmassMotionCfg,
    )
  except Exception:
    return None

  configured_root_str = str(getattr(AmassMotionCfg, "path", "")).strip()
  if not configured_root_str:
    return None
  configured_root = Path(configured_root_str).expanduser()

  yaml_path_str = getattr(AmassMotionCfg, "filtered_motion_selection_filepath", None)
  candidate_paths: list[Path] = []

  if yaml_path_str is not None and str(yaml_path_str).strip():
    yaml_path = Path(str(yaml_path_str)).expanduser()
    if yaml_path.exists() and yaml_path.is_file():
      try:
        with yaml_path.open("r", encoding="utf-8") as file:
          loaded_yaml = yaml.safe_load(file) or {}
      except (yaml.YAMLError, OSError):
        yaml_data = {}
      else:
        yaml_data = loaded_yaml if isinstance(loaded_yaml, dict) else {}
      selected_files = yaml_data.get("selected_files", [])
      if isinstance(selected_files, list):
        for selected_file in selected_files:
          selected_text = str(selected_file).strip()
          if not selected_text:
            continue
          selected_path = Path(selected_text).expanduser()
          if selected_path.is_absolute():
            candidate_paths.append(selected_path)
          else:
            candidate_paths.append(configured_root / selected_path)
            candidate_paths.extend(configured_root.rglob(selected_path.name))

  for pattern in ("**/motion.npz", "**/*parkour*motion*.npz", "**/*parkour*.npz"):
    candidate_paths.extend(configured_root.glob(pattern))

  seen: set[Path] = set()
  for path in candidate_paths:
    resolved_path = path.expanduser().resolve()
    if resolved_path in seen:
      continue
    seen.add(resolved_path)
    if not resolved_path.exists() or not resolved_path.is_file():
      continue
    try:
      validate_tracking_motion_file(resolved_path)
    except (ValueError, OSError, FileNotFoundError):
      continue
    return resolved_path
  return None


def find_default_tracking_motion_file(task_id: str) -> Path | None:
  """Best-effort search for a usable tracking motion file from datasets root."""
  if "Parkour" in task_id:
    from_task_cfg = _find_parkour_motion_from_task_cfg()
    if from_task_cfg is not None:
      return from_task_cfg

  datasets_root = resolve_datasets_root()
  if not datasets_root.exists() or not datasets_root.is_dir():
    return None

  candidate_patterns: list[str] = ["**/motion.npz"]
  if "Parkour" in task_id:
    candidate_patterns = [
      "**/*parkour*motion*.npz",
      "**/*parkour*.npz",
      *candidate_patterns,
    ]

  for pattern in candidate_patterns:
    for path in sorted(datasets_root.glob(pattern)):
      if not path.is_file():
        continue
      try:
        validate_tracking_motion_file(path)
      except (ValueError, OSError, FileNotFoundError):
        continue
      return path.resolve()
  return None


def prepare_distillation_algorithm_cfg(
  *,
  agent_cfg: dict[str, Any],
  obs_format: dict[str, dict[str, tuple[int, ...]]],
  num_actions: int,
  num_rewards: int,
) -> None:
  """Patch teacher policy runtime fields required by TPPO/VaeDistill."""
  algorithm_cfg = agent_cfg.get("algorithm")
  if not isinstance(algorithm_cfg, dict):
    return

  class_name = str(algorithm_cfg.get("class_name", ""))
  if class_name not in {"TPPO", "VaeDistill"}:
    return

  teacher_policy = algorithm_cfg.setdefault("teacher_policy", {})
  if not isinstance(teacher_policy, dict):
    raise TypeError("algorithm.teacher_policy must be a dict for TPPO/VaeDistill")

  label_with_critic = bool(algorithm_cfg.get("label_action_with_critic_obs", True))
  source_group = "critic" if label_with_critic and "critic" in obs_format else "policy"
  if source_group not in obs_format:
    raise KeyError(
      f"Observation group '{source_group}' not found in env obs format. "
      f"Available groups: {sorted(obs_format.keys())}"
    )

  if "obs_format" not in teacher_policy:
    group_format = deepcopy(obs_format[source_group])
    teacher_policy["obs_format"] = {
      "policy": group_format,
      "critic": deepcopy(group_format),
    }
  else:
    teacher_obs_format = teacher_policy["obs_format"]
    if not isinstance(teacher_obs_format, dict):
      raise TypeError("algorithm.teacher_policy.obs_format must be a dict for TPPO/VaeDistill")

    normalized_obs_format: dict[str, dict[str, tuple[int, ...]]] = {}
    for group_name, group_format in teacher_obs_format.items():
      if not isinstance(group_format, dict):
        raise TypeError(
          "algorithm.teacher_policy.obs_format must map group -> dict, "
          f"got group '{group_name}' with value {type(group_format).__name__}"
        )
      normalized_obs_format[group_name] = _normalize_obs_format_shapes(group_format)
    teacher_policy["obs_format"] = normalized_obs_format

  teacher_policy.setdefault("num_actions", num_actions)
  teacher_policy.setdefault("num_rewards", num_rewards)


def _normalize_obs_format_shapes(
  obs_group_format: dict[str, Any],
) -> dict[str, tuple[int, ...]]:
  normalized: dict[str, tuple[int, ...]] = {}
  for term_name, shape in obs_group_format.items():
    if isinstance(shape, tuple):
      normalized[term_name] = tuple(int(dim) for dim in shape)
      continue
    if isinstance(shape, list):
      normalized[term_name] = tuple(int(dim) for dim in shape)
      continue
    try:
      normalized[term_name] = tuple(int(dim) for dim in shape)
    except TypeError as exc:
      raise TypeError(
        f"Invalid obs shape for term '{term_name}': {shape!r}. "
        "Expected tuple/list-like shape."
      ) from exc
  return normalized


def _is_distillation_algorithm(algorithm_cfg: dict[str, Any]) -> bool:
  class_name = str(algorithm_cfg.get("class_name", ""))
  return class_name in {"TPPO", "VaeDistill"}


def _latest_teacher_checkpoint(teacher_logdir: Path) -> Path:
  checkpoints = sorted(
    teacher_logdir.glob("model_*.pt"),
    key=lambda path: int(path.stem.split("_")[1]) if "_" in path.stem else -1,
  )
  if not checkpoints:
    raise FileNotFoundError(
      "teacher_logdir 内未找到 checkpoint（期望文件名形如 model_123.pt）。"
    )
  return checkpoints[-1]


def validate_distillation_runtime_cfg(
  *,
  agent_cfg: dict[str, Any],
  obs_format: dict[str, dict[str, tuple[int, ...]]],
  num_actions: int,
  num_rewards: int,
) -> None:
  """Validate teacher runtime fields after `prepare_distillation_algorithm_cfg`."""
  algorithm_cfg = agent_cfg.get("algorithm")
  if not isinstance(algorithm_cfg, dict) or not _is_distillation_algorithm(algorithm_cfg):
    return

  teacher_policy = algorithm_cfg.get("teacher_policy")
  if not isinstance(teacher_policy, dict):
    raise TypeError("algorithm.teacher_policy must be a dict for TPPO/VaeDistill")

  label_with_critic = bool(algorithm_cfg.get("label_action_with_critic_obs", True))
  source_group = "critic" if label_with_critic and "critic" in obs_format else "policy"
  if source_group not in obs_format:
    raise KeyError(
      f"Observation group '{source_group}' not found in env obs format. "
      f"Available groups: {sorted(obs_format.keys())}"
    )

  teacher_obs_format = teacher_policy.get("obs_format")
  if not isinstance(teacher_obs_format, dict):
    raise ValueError("algorithm.teacher_policy.obs_format is required for TPPO/VaeDistill")
  teacher_policy_obs_format = teacher_obs_format.get("policy")
  if not isinstance(teacher_policy_obs_format, dict):
    raise ValueError("algorithm.teacher_policy.obs_format.policy must be a dict")
  teacher_policy_obs_format = _normalize_obs_format_shapes(teacher_policy_obs_format)
  runtime_obs_format = _normalize_obs_format_shapes(obs_format[source_group])
  if teacher_policy_obs_format != runtime_obs_format:
    raise ValueError(
      "teacher_policy.obs_format.policy 与当前环境观测不一致。"
      f"期望来源组: {source_group}，"
      f"runtime={runtime_obs_format}，teacher={teacher_policy_obs_format}。"
      "可删除 teacher_policy.obs_format 让脚本自动注入。"
    )

  teacher_num_actions = teacher_policy.get("num_actions")
  teacher_num_rewards = teacher_policy.get("num_rewards")
  if teacher_num_actions != num_actions:
    raise ValueError(
      "teacher_policy.num_actions 与环境动作维度不一致："
      f"teacher={teacher_num_actions}, env={num_actions}"
    )
  if teacher_num_rewards != num_rewards:
    raise ValueError(
      "teacher_policy.num_rewards 与环境奖励维度不一致："
      f"teacher={teacher_num_rewards}, env={num_rewards}"
    )


def validate_distillation_teacher_assets(*, agent_cfg: dict[str, Any]) -> Path | None:
  """Validate `teacher_logdir` and return latest checkpoint path when required."""
  algorithm_cfg = agent_cfg.get("algorithm")
  if not isinstance(algorithm_cfg, dict) or not _is_distillation_algorithm(algorithm_cfg):
    return None

  teacher_logdir = algorithm_cfg.get("teacher_logdir")
  if teacher_logdir is None or str(teacher_logdir).strip() == "":
    raise ValueError(
      "当前算法需要 teacher 权重。请显式传入：\n"
      "  --agent.algorithm.teacher-logdir /path/to/teacher_run"
    )

  teacher_dir = Path(str(teacher_logdir)).expanduser().resolve()
  if not teacher_dir.exists() or not teacher_dir.is_dir():
    raise FileNotFoundError(f"teacher_logdir 不存在或不是目录: {teacher_dir}")

  latest_checkpoint = _latest_teacher_checkpoint(teacher_dir)
  checkpoint_data = torch.load(str(latest_checkpoint), map_location="cpu")
  if not isinstance(checkpoint_data, dict) or "model_state_dict" not in checkpoint_data:
    raise ValueError(
      f"teacher checkpoint 格式非法（缺少 model_state_dict）: {latest_checkpoint}"
    )

  if "policy_normalizer_state_dict" in checkpoint_data:
    agent_yaml_path = teacher_dir / "params" / "agent.yaml"
    if not agent_yaml_path.exists():
      raise FileNotFoundError(
        "teacher checkpoint 含有 policy_normalizer_state_dict，"
        f"但缺少 {agent_yaml_path}"
      )

    with agent_yaml_path.open("r", encoding="utf-8") as file:
      teacher_agent_cfg = yaml.safe_load(file)
    normalizers_cfg = teacher_agent_cfg.get("normalizers", {})
    policy_normalizer_cfg = normalizers_cfg.get("policy")
    if not isinstance(policy_normalizer_cfg, dict):
      raise ValueError(
        "teacher params/agent.yaml 缺少 normalizers.policy 配置，"
        "无法构建 teacher normalizer。"
      )
    if "class_name" not in policy_normalizer_cfg:
      raise ValueError(
        "teacher params/agent.yaml 中 normalizers.policy 缺少 class_name。"
      )

  return latest_checkpoint
