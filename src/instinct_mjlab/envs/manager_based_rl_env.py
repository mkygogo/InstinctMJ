from __future__ import annotations

from collections.abc import Sequence

import torch

from mjlab.envs import ManagerBasedRlEnv

from instinct_mjlab.managers import MultiRewardCfg, MultiRewardManager
from instinct_mjlab.monitors import MonitorManager


class InstinctRlEnv(ManagerBasedRlEnv):
  """This class adds additional logging mechanism on sensors to get more
  comprehensive running statistics.
  """

  def load_managers(self) -> None:
    # check and routing the reward manager to the multi reward manager
    reward_group_cfg = None
    if isinstance(self.cfg.rewards, MultiRewardCfg):
      reward_group_cfg = self.cfg.rewards
      self.cfg.rewards = {}

    super().load_managers()

    # replace the parent class's reward manager
    if reward_group_cfg is not None:
      self.cfg.rewards = reward_group_cfg
      self.reward_manager = MultiRewardManager(
        self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
      )

    monitor_cfg = getattr(self.cfg, "monitors", None)
    if monitor_cfg is None:
      monitor_cfg = {}
    self.monitor_manager = MonitorManager(monitor_cfg, self)

  def setup_manager_visualizers(self) -> None:
    super().setup_manager_visualizers()
    if getattr(self, "monitor_manager", None) is not None:
      self.manager_visualizers["monitor_manager"] = self.monitor_manager

  def step(self, action: torch.Tensor):
    obs, reward, terminated, truncated, extras = super().step(action)
    if getattr(self, "monitor_manager", None) is not None:
      monitor_infos = self.monitor_manager.update(dt=self.step_dt)
      extras.setdefault("step", {})
      extras["step"].update(monitor_infos)
    return obs, reward, terminated, truncated, extras

  def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
    if isinstance(env_ids, Sequence):
      env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)

    super()._reset_idx(env_ids)

    if getattr(self, "monitor_manager", None) is not None:
      monitor_infos = self.monitor_manager.reset(env_ids, is_episode=True)
      self.extras["log"] = self.extras.get("log", {})
      self.extras["log"].update(monitor_infos)

  """
  Properties.
  """

  @property
  def num_rewards(self) -> int:
    return getattr(self.reward_manager, "num_rewards", 1)
