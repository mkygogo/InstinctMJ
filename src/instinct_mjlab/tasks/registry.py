"""Task registry for Instinct-RL based mjlab tasks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from instinct_mjlab.utils.deepcopy_compat import patch_deepcopy_for_abcmeta

from instinct_mjlab.rl import InstinctRlOnPolicyRunnerCfg
from mjlab.envs import ManagerBasedRlEnvCfg


# Ensure ABCMeta deepcopy fix is applied before any deepcopy calls.
patch_deepcopy_for_abcmeta()


@dataclass
class _TaskCfg:
  env_cfg: ManagerBasedRlEnvCfg
  play_env_cfg: ManagerBasedRlEnvCfg
  instinct_rl_cfg: InstinctRlOnPolicyRunnerCfg
  runner_cls: type | None


_REGISTRY: dict[str, _TaskCfg] = {}


def register_instinct_task(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg,
  play_env_cfg: ManagerBasedRlEnvCfg,
  instinct_rl_cfg: InstinctRlOnPolicyRunnerCfg,
  runner_cls: type | None = None,
) -> None:
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered.")
  _REGISTRY[task_id] = _TaskCfg(env_cfg, play_env_cfg, instinct_rl_cfg, runner_cls)


def list_tasks() -> list[str]:
  return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  task_cfg = _REGISTRY[task_name]
  return deepcopy(task_cfg.play_env_cfg if play else task_cfg.env_cfg)


def load_instinct_rl_cfg(task_name: str) -> InstinctRlOnPolicyRunnerCfg:
  return deepcopy(_REGISTRY[task_name].instinct_rl_cfg)


def load_runner_cls(task_name: str) -> type | None:
  return _REGISTRY[task_name].runner_cls
