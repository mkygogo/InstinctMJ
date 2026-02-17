from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_lin_vel_xy_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking reference linear velocity (x/y in anchor frame)."""
  command_term = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  if hasattr(command_term, "anchor_quat_w") and hasattr(command_term, "anchor_lin_vel_w"):
    ref_lin_vel_b = quat_apply_inverse(command_term.anchor_quat_w, command_term.anchor_lin_vel_w)
    if hasattr(command_term, "robot_anchor_lin_vel_w"):
      robot_lin_vel_b = quat_apply_inverse(command_term.anchor_quat_w, command_term.robot_anchor_lin_vel_w)
    else:
      robot_lin_vel_b = asset.data.root_link_lin_vel_b
  else:
    command = env.command_manager.get_command(command_name)
    if command.shape[1] < 3:
      raise ValueError(
        f"Command '{command_name}' must have at least 3 dims (vx, vy, wz), got {command.shape[1]}."
      )
    ref_lin_vel_b = torch.zeros((env.num_envs, 3), device=env.device, dtype=command.dtype)
    ref_lin_vel_b[:, :2] = command[:, :2]
    robot_lin_vel_b = asset.data.root_link_lin_vel_b

  error = torch.sum(torch.square(ref_lin_vel_b[:, :2] - robot_lin_vel_b[:, :2]), dim=1)
  return torch.exp(-error / max(std, 1e-6) ** 2)


def track_ang_vel_z_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking reference yaw angular velocity (anchor frame)."""
  command_term = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  if hasattr(command_term, "anchor_quat_w") and hasattr(command_term, "anchor_ang_vel_w"):
    ref_ang_vel_b = quat_apply_inverse(command_term.anchor_quat_w, command_term.anchor_ang_vel_w)
    if hasattr(command_term, "robot_anchor_ang_vel_w"):
      robot_ang_vel_b = quat_apply_inverse(command_term.anchor_quat_w, command_term.robot_anchor_ang_vel_w)
    else:
      robot_ang_vel_b = asset.data.root_link_ang_vel_b
  else:
    command = env.command_manager.get_command(command_name)
    if command.shape[1] < 3:
      raise ValueError(
        f"Command '{command_name}' must have at least 3 dims (vx, vy, wz), got {command.shape[1]}."
      )
    ref_ang_vel_b = torch.zeros((env.num_envs, 3), device=env.device, dtype=command.dtype)
    ref_ang_vel_b[:, 2] = command[:, 2]
    robot_ang_vel_b = asset.data.root_link_ang_vel_b

  error = torch.square(ref_ang_vel_b[:, 2] - robot_ang_vel_b[:, 2])
  return torch.exp(-error / max(std, 1e-6) ** 2)


def heading_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Compute heading command magnitude (InstinctLab-compatible)."""
  command_term = env.command_manager.get_term(command_name)
  if hasattr(command_term, "anchor_quat_w") and hasattr(command_term, "anchor_ang_vel_w"):
    ref_ang_vel_b = quat_apply_inverse(command_term.anchor_quat_w, command_term.anchor_ang_vel_w)
    return torch.abs(ref_ang_vel_b[:, 2])

  command = env.command_manager.get_command(command_name)
  if command.shape[1] < 3:
    raise ValueError(
      f"Command '{command_name}' must have at least 3 dims (vx, vy, wz), got {command.shape[1]}."
    )
  return torch.abs(command[:, 2])


def dont_wait(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize standing still when there is a forward velocity command."""
  command_term = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  if hasattr(command_term, "anchor_quat_w") and hasattr(command_term, "anchor_lin_vel_w"):
    ref_lin_vel_b = quat_apply_inverse(command_term.anchor_quat_w, command_term.anchor_lin_vel_w)
    if hasattr(command_term, "robot_anchor_lin_vel_w"):
      robot_lin_vel_b = quat_apply_inverse(command_term.anchor_quat_w, command_term.robot_anchor_lin_vel_w)
    else:
      robot_lin_vel_b = asset.data.root_link_lin_vel_b
    lin_vel_cmd_x = ref_lin_vel_b[:, 0]
    lin_vel_x = robot_lin_vel_b[:, 0]
  else:
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_link_lin_vel_b[:, 0]

  return (lin_vel_cmd_x > 0.3) * (
    (lin_vel_x < 0.15).float() + (lin_vel_x < 0.0).float() + (lin_vel_x < -0.15).float()
  )


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  threshold: float = 0.15,
  offset: float = 1.0,
) -> torch.Tensor:
  """Penalize moving when there is no velocity command."""
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  dof_error = torch.sum(torch.abs(asset.data.joint_pos - default_joint_pos), dim=1)

  command_term = env.command_manager.get_term(command_name)
  if hasattr(command_term, "anchor_quat_w") and hasattr(command_term, "anchor_lin_vel_w"):
    ref_lin_vel_b = quat_apply_inverse(command_term.anchor_quat_w, command_term.anchor_lin_vel_w)
    ref_ang_vel_b = quat_apply_inverse(command_term.anchor_quat_w, command_term.anchor_ang_vel_w)
    cmd_lin_norm = torch.norm(ref_lin_vel_b[:, :2], dim=1)
    cmd_yaw_abs = torch.abs(ref_ang_vel_b[:, 2])
  else:
    cmd = env.command_manager.get_command(command_name)
    if cmd.shape[1] < 3:
      raise ValueError(
        f"Command '{command_name}' must have at least 3 dims (vx, vy, wz), got {cmd.shape[1]}."
      )
    cmd_lin_norm = torch.norm(cmd[:, :2], dim=1)
    cmd_yaw_abs = torch.abs(cmd[:, 2])

  return (dof_error - offset) * (cmd_lin_norm < threshold) * (cmd_yaw_abs < threshold)


def feet_air_time(
  env: ManagerBasedRlEnv,
  command_name: str,
  vel_threshold: float,
  sensor_cfg: SceneEntityCfg | None = None,
  sensor_name: str | None = None,
) -> torch.Tensor:
  """Reward long steps taken by the feet for bipeds."""
  if sensor_name is None:
    if sensor_cfg is None:
      raise ValueError("Either sensor_name or sensor_cfg must be provided.")
    sensor_name = sensor_cfg.name

  contact_sensor: ContactSensor = env.scene[sensor_name]
  air_time = contact_sensor.data.current_air_time
  contact_time = contact_sensor.data.current_contact_time
  if air_time is None or contact_time is None:
    raise ValueError(f"Sensor '{sensor_name}' must enable track_air_time for feet_air_time reward.")

  if sensor_cfg is not None:
    air_time = air_time[:, sensor_cfg.body_ids]
    contact_time = contact_time[:, sensor_cfg.body_ids]

  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.sum(in_contact.int(), dim=1) == 1
  reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]

  command_term = env.command_manager.get_term(command_name)
  if hasattr(command_term, "anchor_quat_w") and hasattr(command_term, "anchor_lin_vel_w"):
    ref_lin_vel_b = quat_apply_inverse(command_term.anchor_quat_w, command_term.anchor_lin_vel_w)
    ref_ang_vel_b = quat_apply_inverse(command_term.anchor_quat_w, command_term.anchor_ang_vel_w)
    cmd_lin_norm = torch.norm(ref_lin_vel_b[:, :2], dim=1)
    cmd_yaw_abs = torch.abs(ref_ang_vel_b[:, 2])
  else:
    cmd = env.command_manager.get_command(command_name)
    if cmd.shape[1] < 3:
      raise ValueError(
        f"Command '{command_name}' must have at least 3 dims (vx, vy, wz), got {cmd.shape[1]}."
      )
    cmd_lin_norm = torch.norm(cmd[:, :2], dim=1)
    cmd_yaw_abs = torch.abs(cmd[:, 2])

  reward *= torch.logical_or(cmd_lin_norm > vel_threshold, cmd_yaw_abs > vel_threshold)
  return reward


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  threshold: float = 0.1,
) -> torch.Tensor:
  """Penalize foot sliding speed while feet are in contact."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  if found is None:
    return torch.zeros(env.num_envs, device=env.device)

  if found.ndim == 3:
    in_contact = torch.any(found > 0, dim=-1)
  else:
    in_contact = found > 0

  body_vel_w = getattr(asset.data, "body_link_lin_vel_w", None)
  if body_vel_w is None:
    body_vel_w = getattr(asset.data, "body_lin_vel_w", None)
  if body_vel_w is None:
    return torch.zeros(env.num_envs, device=env.device)

  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice):
    body_ids = list(range(body_vel_w.shape[1]))[body_ids]
  else:
    body_ids = list(body_ids)
  if len(body_ids) == 0:
    return torch.zeros(env.num_envs, device=env.device)

  foot_vel_xy = body_vel_w[:, body_ids, :2]
  slip_speed = torch.norm(foot_vel_xy, dim=-1)
  slip_penalty = torch.clamp(slip_speed - threshold, min=0.0)
  return torch.sum(slip_penalty * in_contact.float(), dim=1)


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def joint_deviation_square(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(joint_error), dim=1)


def joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(joint_error), dim=1)


def link_orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat link orientation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  body_quat_w = getattr(asset.data, "body_link_quat_w", None)
  if body_quat_w is None:
    body_quat_w = getattr(asset.data, "body_quat_w", None)
  if body_quat_w is None:
    raise AttributeError("Robot data is missing body_link_quat_w/body_quat_w for orientation terms.")

  link_quat = body_quat_w[:, asset_cfg.body_ids[0], :]
  link_projected_gravity = quat_apply_inverse(link_quat, asset.data.gravity_vec_w)
  return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)


def feet_orientation_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  contact_force_threshold: float = 1.0,
) -> torch.Tensor:
  """Reward feet being oriented vertically when in contact with the ground."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]

  body_quat_w = getattr(asset.data, "body_link_quat_w", None)
  if body_quat_w is None:
    body_quat_w = getattr(asset.data, "body_quat_w", None)
  if body_quat_w is None:
    raise AttributeError("Robot data is missing body_link_quat_w/body_quat_w for orientation terms.")

  body_quat_w = body_quat_w[:, asset_cfg.body_ids, :]
  num_envs, num_feet = body_quat_w.shape[:2]

  gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, num_feet, -1)
  projected_gravity = quat_apply_inverse(
    body_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3)
  ).reshape(num_envs, num_feet, 3)
  orientation_error = torch.linalg.vector_norm(projected_gravity[:, :, :2], dim=-1)

  if contact_sensor.data.force is not None:
    in_contact = torch.linalg.vector_norm(contact_sensor.data.force, dim=-1) > contact_force_threshold
  elif contact_sensor.data.found is not None:
    in_contact = contact_sensor.data.found > 0
  else:
    return torch.zeros(env.num_envs, device=env.device)

  if in_contact.ndim == 1:
    in_contact = in_contact.unsqueeze(-1)
  in_contact = in_contact[:, :num_feet]
  return torch.sum(orientation_error * in_contact.float(), dim=1)


def feet_at_plane(
  env: ManagerBasedRlEnv,
  contact_sensor_name: str | None = None,
  left_height_scanner_name: str | None = None,
  right_height_scanner_name: str | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  height_offset: float = 0.035,
  contact_force_threshold: float = 1.0,
  contact_sensor_cfg: SceneEntityCfg | None = None,
  left_height_scanner_cfg: SceneEntityCfg | None = None,
  right_height_scanner_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Reward feet being at certain height above the ground plane."""
  if contact_sensor_name is None:
    if contact_sensor_cfg is None:
      raise ValueError("Either contact_sensor_name or contact_sensor_cfg must be provided.")
    contact_sensor_name = contact_sensor_cfg.name
  if left_height_scanner_name is None:
    if left_height_scanner_cfg is None:
      raise ValueError("Either left_height_scanner_name or left_height_scanner_cfg must be provided.")
    left_height_scanner_name = left_height_scanner_cfg.name
  if right_height_scanner_name is None:
    if right_height_scanner_cfg is None:
      raise ValueError("Either right_height_scanner_name or right_height_scanner_cfg must be provided.")
    right_height_scanner_name = right_height_scanner_cfg.name

  asset: Entity = env.scene[asset_cfg.name]
  body_pos_w = getattr(asset.data, "body_link_pos_w", None)
  if body_pos_w is None:
    body_pos_w = getattr(asset.data, "body_pos_w", None)
  if body_pos_w is None:
    body_pos_w = getattr(asset.data, "body_com_pos_w", None)
  if body_pos_w is None:
    raise AttributeError("Robot data is missing body_link_pos_w/body_pos_w/body_com_pos_w for foot terms.")

  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice):
    body_ids = list(range(body_pos_w.shape[1]))[body_ids]
  else:
    body_ids = list(body_ids)
  if len(body_ids) < 2:
    return torch.zeros(env.num_envs, device=env.device)

  contact_sensor: ContactSensor = env.scene[contact_sensor_name]
  if contact_sensor.data.force is not None:
    is_contact = torch.linalg.vector_norm(contact_sensor.data.force, dim=-1) > contact_force_threshold
  elif contact_sensor.data.found is not None:
    is_contact = contact_sensor.data.found > 0
  else:
    is_contact = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
  if is_contact.ndim == 1:
    is_contact = is_contact.unsqueeze(-1)

  left_sensor: RayCastSensor = env.scene[left_height_scanner_name]
  right_sensor: RayCastSensor = env.scene[right_height_scanner_name]
  left_hit_z = left_sensor.data.hit_pos_w[..., 2]
  right_hit_z = right_sensor.data.hit_pos_w[..., 2]
  left_hit_z = torch.where(left_sensor.data.distances < 0.0, 0.0, left_hit_z)
  right_hit_z = torch.where(right_sensor.data.distances < 0.0, 0.0, right_hit_z)

  left_height = body_pos_w[:, body_ids[0], 2].unsqueeze(-1)
  right_height = body_pos_w[:, body_ids[1], 2].unsqueeze(-1)

  left_contact = is_contact[:, 0:1].float()
  right_contact = is_contact[:, 1:2].float() if is_contact.shape[1] > 1 else 0.0

  left_reward = torch.clamp(left_height - left_hit_z - height_offset, min=0.0, max=0.3) * left_contact
  right_reward = torch.clamp(right_height - right_hit_z - height_offset, min=0.0, max=0.3) * right_contact
  return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def feet_close_xy_gauss(
  env: ManagerBasedRlEnv,
  threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  std: float = 0.1,
) -> torch.Tensor:
  """Penalize when feet are too close together in the y distance."""
  asset: Entity = env.scene[asset_cfg.name]
  body_pos_w = getattr(asset.data, "body_link_pos_w", None)
  if body_pos_w is None:
    body_pos_w = getattr(asset.data, "body_pos_w", None)
  if body_pos_w is None:
    body_pos_w = getattr(asset.data, "body_com_pos_w", None)
  if body_pos_w is None:
    raise AttributeError("Robot data is missing body_link_pos_w/body_pos_w/body_com_pos_w for foot terms.")

  body_pos_w = body_pos_w[:, asset_cfg.body_ids, :]
  if body_pos_w.shape[1] < 2:
    return torch.zeros(env.num_envs, device=env.device)

  left_foot_xy = body_pos_w[:, 0, :2]
  right_foot_xy = body_pos_w[:, 1, :2]
  heading_w = asset.data.heading_w

  cos_heading = torch.cos(heading_w)
  sin_heading = torch.sin(heading_w)

  left_y = -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1]
  right_y = -sin_heading * right_foot_xy[:, 0] + cos_heading * right_foot_xy[:, 1]
  feet_distance_y = torch.abs(left_y - right_y)

  return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def volume_points_penetration(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  tolerance: float = 0.0,
) -> torch.Tensor:
  sensor = env.scene.sensors[sensor_name]
  penetration = sensor.data.penetration_offset
  points_vel = sensor.data.points_vel_w

  penetration_depth = torch.linalg.vector_norm(penetration.reshape(env.num_envs, -1, 3), dim=-1)
  in_obstacle = (penetration_depth > tolerance).float()
  points_vel_norm = torch.linalg.vector_norm(points_vel.reshape(env.num_envs, -1, 3), dim=-1)
  velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth
  return torch.sum(velocity_times_penetration, dim=-1)


def motors_power_square(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  normalize_by_stiffness: bool = True,
  normalize_by_num_joints: bool = False,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torque = getattr(asset.data, "applied_torque", None)
  if torque is None:
    torque = getattr(asset.data, "actuator_force", None)
  if torque is None:
    torque = torch.zeros_like(asset.data.joint_vel, device=env.device)
  elif torque.shape != asset.data.joint_vel.shape:
    matched_torque = torch.zeros_like(asset.data.joint_vel, device=env.device)
    num_cols = min(matched_torque.shape[1], torque.shape[1])
    matched_torque[:, :num_cols] = torque[:, :num_cols]
    torque = matched_torque

  power_j = torque * asset.data.joint_vel
  if normalize_by_stiffness:
    actuators = asset.actuators.values() if isinstance(asset.actuators, dict) else asset.actuators
    for actuator in actuators:
      stiffness = getattr(actuator, "stiffness", None)
      joint_indices = getattr(actuator, "joint_indices", None)
      if stiffness is None or joint_indices is None:
        continue
      joint_indices = torch.as_tensor(joint_indices, device=env.device, dtype=torch.long)
      if joint_indices.numel() == 0:
        continue

      if torch.is_tensor(stiffness):
        if stiffness.ndim == 2:
          stiffness_values = stiffness[:, :joint_indices.numel()]
        else:
          stiffness_values = stiffness[:joint_indices.numel()]
      else:
        stiffness_values = torch.as_tensor(stiffness, device=env.device, dtype=power_j.dtype)

      power_j[:, joint_indices] = power_j[:, joint_indices] / torch.clamp(stiffness_values, min=1e-6)

  power_j = power_j[:, asset_cfg.joint_ids]
  power = torch.sum(torch.square(power_j), dim=-1)
  if normalize_by_num_joints and power_j.shape[-1] > 0:
    power = power / power_j.shape[-1]
  return power


def joint_vel_limits(
  env: ManagerBasedRlEnv,
  soft_ratio: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  joint_vel_limits = getattr(asset.data, "joint_vel_limits", None)
  if joint_vel_limits is None:
    joint_vel_limits = getattr(asset.data, "joint_velocity_limits", None)
  if joint_vel_limits is None:
    return torch.zeros(env.num_envs, device=env.device)
  if joint_vel_limits.ndim == 1:
    joint_vel_limits = joint_vel_limits.unsqueeze(0).expand_as(asset.data.joint_vel)

  vel_abs = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
  soft_limits = joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
  out_of_limits = torch.clamp(vel_abs - soft_limits, min=0.0)
  return torch.sum(out_of_limits, dim=1)


def applied_torque_limits_by_ratio(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  limit_ratio: float = 0.8,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  joint_effort_limits = getattr(asset.data, "joint_effort_limits", None)
  if joint_effort_limits is None:
    return torch.zeros(env.num_envs, device=env.device)

  joint_effort_limits = joint_effort_limits[:, asset_cfg.joint_ids]
  applied_torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
  out_of_limits = torch.clamp(applied_torque - joint_effort_limits * limit_ratio, min=0.0)
  return torch.sum(torch.square(out_of_limits), dim=-1)


def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]
  if contact_sensor.data.force is not None:
    is_contact = torch.linalg.vector_norm(contact_sensor.data.force, dim=-1) > threshold
  elif contact_sensor.data.found is not None:
    is_contact = contact_sensor.data.found > 0
  else:
    return torch.zeros(env.num_envs, device=env.device)

  if is_contact.ndim == 1:
    is_contact = is_contact.unsqueeze(-1)
  return torch.sum(is_contact.float(), dim=1)
