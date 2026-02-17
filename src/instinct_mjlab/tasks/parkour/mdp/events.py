from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import mjlab.envs.mdp as envs_mdp
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, sample_uniform

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def randomize_rigid_body_material(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  static_friction_range: tuple[float, float] = (0.3, 1.0),
  dynamic_friction_range: tuple[float, float] = (0.3, 1.0),
  restitution_range: tuple[float, float] = (0.0, 0.0),
  num_buckets: int = 64,
  make_consistent: bool = True,
) -> None:
  del num_buckets
  envs_mdp.randomize_field(
    env=env,
    env_ids=env_ids,
    field="geom_friction",
    ranges={
      0: static_friction_range,
      1: dynamic_friction_range,
      2: restitution_range,
    },
    operation="abs",
    asset_cfg=asset_cfg,
    shared_random=make_consistent,
  )


def push_by_setting_velocity_without_stand(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  velocity_range: dict[str, tuple[float, float]],
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  velocity_threshold: float = 0.15,
) -> None:
  """Push the asset by setting root velocity. No push when standing still."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if env_ids.numel() == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids]

  range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
  ranges = torch.tensor(range_list, device=env.device)
  add_vel = sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)

  command_term = env.command_manager.get_term(command_name)
  if hasattr(command_term, "anchor_quat_w") and hasattr(command_term, "anchor_lin_vel_w"):
    ref_lin_vel_b = quat_apply_inverse(command_term.anchor_quat_w, command_term.anchor_lin_vel_w)
    ref_ang_vel_b = quat_apply_inverse(command_term.anchor_quat_w, command_term.anchor_ang_vel_w)
    lin_vel = torch.norm(ref_lin_vel_b[env_ids, :2], dim=1) > velocity_threshold
    ang_vel = torch.abs(ref_ang_vel_b[env_ids, 2]) > velocity_threshold
  else:
    cmd = env.command_manager.get_command(command_name)[env_ids]
    lin_vel = torch.norm(cmd[:, :2], dim=1) > velocity_threshold
    ang_vel = torch.abs(cmd[:, 2]) > velocity_threshold

  should_push = torch.logical_or(lin_vel, ang_vel).float().unsqueeze(-1)
  vel_w += add_vel * should_push
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)
