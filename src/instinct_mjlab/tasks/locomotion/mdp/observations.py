"""Common functions that can be used to define observations for locomotion."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg


def joint_vel(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Joint velocity in the articulation frame."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids]
