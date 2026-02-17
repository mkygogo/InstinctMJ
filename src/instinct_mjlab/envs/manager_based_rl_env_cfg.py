from __future__ import annotations

from dataclasses import dataclass

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg

from instinct_mjlab.envs.ui import InstinctLabRLEnvWindow


@dataclass
class InstinctLabRLEnvCfg(ManagerBasedRlEnvCfg):
  """Configuration for a reinforcement learning environment with the manager-based workflow."""

  # ui settings
  ui_window_class_type: type | None = InstinctLabRLEnvWindow
  """Inherit from the manager-based RL environment window class."""

  # monitor settings
  monitors: object | None = None
  """Monitor Settings.

  Please refer to the `instinct_mjlab.monitors.MonitorManager` class for more details.
  """
