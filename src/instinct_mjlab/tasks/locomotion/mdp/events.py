# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define events for the learning environment."""

from __future__ import annotations

from typing import Literal

import torch

import mjlab.envs.mdp as mdp
from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def randomize_rigid_body_material(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  static_friction_range: tuple[float, float] = (0.3, 1.0),
  dynamic_friction_range: tuple[float, float] = (0.3, 1.0),
  restitution_range: tuple[float, float] = (0.0, 0.0),
  num_buckets: int = 64,
) -> None:
  """Randomize rigid-body geom friction coefficients."""
  del num_buckets
  mdp.dr.geom_friction(
    env=env,
    env_ids=env_ids,
    ranges={
      0: static_friction_range,
      1: dynamic_friction_range,
      2: restitution_range,
    },
    operation="abs",
    asset_cfg=asset_cfg,
    axes=[0, 1, 2],
  )


def randomize_rigid_body_mass(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  mass_distribution_params: tuple[float, float] = (-0.5, 0.5),
  operation: Literal["add", "scale", "abs"] = "add",
) -> None:
  """Randomize rigid-body mass."""
  mdp.dr.body_mass(
    env=env,
    env_ids=env_ids,
    ranges=mass_distribution_params,
    operation=operation,
    asset_cfg=asset_cfg,
  )


def reset_joints_by_scale(
  env,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset joint state by scaling default joint positions and offsetting velocities."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
  joint_pos *= sample_uniform(*position_range, joint_pos.shape, env.device)
  joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )
