from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg

from instinct_mjlab.envs.viewer_cfg import InstinctLabViewerConfig


@dataclass
class InstinctLabRLEnvCfg(ManagerBasedRlEnvCfg):
  """Configuration for a reinforcement learning environment with the manager-based workflow."""

  viewer: InstinctLabViewerConfig = field(default_factory=InstinctLabViewerConfig)
  """Viewer Settings."""

  # monitor settings
  monitors: object | None = None
  """Monitor Settings.

  Please refer to the `instinct_mjlab.monitors.MonitorManager` class for more details.
  """
