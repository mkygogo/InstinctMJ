"""Customized Unitree G1 asset for mjlab migration."""

from __future__ import annotations

import copy
import os
from typing import TypeAlias

import mujoco

from mjlab.actuator import (
  ActuatorCfg,
  BuiltinPositionActuatorCfg,
  DelayedActuatorCfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

# MJCF (XML) path – uses the local 29-dof torso-base popsicle model,
# matching InstinctLab's G1_29DOF_TORSOBASE_POPSICLE_CFG.
G1_MJCF_PATH: str = os.path.join(
  __file_dir__, "resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml"
)
G1_MESHES_DIR: str = os.path.join(__file_dir__, "resources/unitree_g1/meshes")

LimitCfg: TypeAlias = float | dict[str, float]

"""
joint name order:
[
    'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint',
    'waist_pitch_joint',
    'left_shoulder_roll_joint',
    'right_shoulder_roll_joint',
    'waist_roll_joint',
    'left_shoulder_yaw_joint',
    'right_shoulder_yaw_joint',
    'waist_yaw_joint',
    'left_elbow_joint',
    'right_elbow_joint',
    'left_hip_pitch_joint',
    'right_hip_pitch_joint',
    'left_wrist_roll_joint',
    'right_wrist_roll_joint',
    'left_hip_roll_joint',
    'right_hip_roll_joint',
    'left_wrist_pitch_joint',
    'right_wrist_pitch_joint',
    'left_hip_yaw_joint',
    'right_hip_yaw_joint',
    'left_wrist_yaw_joint',
    'right_wrist_yaw_joint',
    'left_knee_joint',
    'right_knee_joint',
    'left_ankle_pitch_joint',
    'right_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
]
"""

G1_29DOF_INSTINCTLAB_JOINT_ORDER: tuple[str, ...] = (
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "waist_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "waist_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "waist_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
)


def _get_popsicle_spec() -> mujoco.MjSpec:
  """Load the local g1_29dof_torsobase_popsicle.xml as MjSpec.

  The free joint is attached to `torso_link`, so root pose/init_state is applied on torso_link.
  """
  return mujoco.MjSpec.from_file(G1_MJCF_PATH)


def get_g1_assets(meshdir: str | None) -> dict[str, bytes]:
  """Load local G1 mesh assets keyed with MuJoCo meshdir prefix."""
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_MESHES_DIR, meshdir)
  return assets


# Initial state matching InstinctLab G1_29DOF_TORSOBASE_CFG (simplified variant).
# NOTE: pos is the root (torso_link) world position.
_SIMPLIFIED_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.82),
  joint_pos={
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    ".*_wrist_roll_joint": 0.0,
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_yaw_joint": 0.0,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.35,
  },
  joint_vel={".*": 0.0},
)

# Initial state matching InstinctLab G1_29DOF_TORSOBASE_POPSICLE_CFG.
# NOTE: pos is the root (torso_link) world position.
_POPSICLE_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.82),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)


def _new_g1_cfg(
  init_state: EntityCfg.InitialStateCfg | None = None,
) -> EntityCfg:
  """Create a fresh G1 robot config using the local popsicle XML."""
  return EntityCfg(
    init_state=copy.deepcopy(init_state or _POPSICLE_INIT_STATE),
    spec_fn=_get_popsicle_spec,
    articulation=EntityArticulationInfoCfg(),
  )


def _set_robot_actuators(
  robot_cfg: EntityCfg,
  actuator_cfgs: tuple[ActuatorCfg, ...],
  *,
  soft_joint_pos_limit_factor: float | None = None,
) -> EntityCfg:
  cfg = copy.deepcopy(robot_cfg)
  if cfg.articulation is None:
    return cfg
  cfg.articulation.actuators = tuple(copy.deepcopy(act) for act in actuator_cfgs)
  if soft_joint_pos_limit_factor is not None:
    cfg.articulation.soft_joint_pos_limit_factor = soft_joint_pos_limit_factor
  return cfg


def _builtin_pos_cfg(
  target_names_expr: tuple[str, ...],
  *,
  effort_limit: float,
  stiffness: float,
  damping: float,
  armature: float,
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    effort_limit=effort_limit,
    stiffness=stiffness,
    damping=damping,
    armature=armature,
  )


def _delayed_pos_cfg(
  base_cfg: BuiltinPositionActuatorCfg,
  *,
  min_delay: int,
  max_delay: int,
) -> DelayedActuatorCfg:
  return DelayedActuatorCfg(
    base_cfg=copy.deepcopy(base_cfg),
    delay_target="position",
    delay_min_lag=min_delay,
    delay_max_lag=max_delay,
  )


def _resolve_limit_value(limit_cfg: LimitCfg, key: str) -> float:
  if isinstance(limit_cfg, dict):
    return float(limit_cfg[key])
  return float(limit_cfg)


# Motor specs (from Unitree), aligned with mjlab's original g1_constants.
ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))
ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))
ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)
ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)

# Motor output limits (aligned with mjlab's original g1_constants).
ACTUATOR_5020_EFFORT_LIMIT = 25.0
ACTUATOR_5020_VELOCITY_LIMIT = 37.0
ACTUATOR_7520_14_EFFORT_LIMIT = 88.0
ACTUATOR_7520_14_VELOCITY_LIMIT = 32.0
ACTUATOR_7520_22_EFFORT_LIMIT = 139.0
ACTUATOR_7520_22_VELOCITY_LIMIT = 20.0
ACTUATOR_4010_EFFORT_LIMIT = 5.0
ACTUATOR_4010_VELOCITY_LIMIT = 22.0
ACTUATOR_DUAL_5020_EFFORT_LIMIT = ACTUATOR_5020_EFFORT_LIMIT * 2.0

# Following the principles of BeyondMimic, and the kp/kd computation logic.
# NOTE: These logic are still being tested, so we put them here for substitution in users Cfg class.
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# Used by parkour joint-velocity-limit reward reconstruction.
G1_JOINT_VEL_LIMIT_BY_EFFORT_LIMIT: dict[float, float] = {
  ACTUATOR_7520_14_EFFORT_LIMIT: ACTUATOR_7520_14_VELOCITY_LIMIT,
  ACTUATOR_7520_22_EFFORT_LIMIT: ACTUATOR_7520_22_VELOCITY_LIMIT,
  ACTUATOR_5020_EFFORT_LIMIT: ACTUATOR_5020_VELOCITY_LIMIT,
  ACTUATOR_DUAL_5020_EFFORT_LIMIT: ACTUATOR_5020_VELOCITY_LIMIT,
  ACTUATOR_4010_EFFORT_LIMIT: ACTUATOR_4010_VELOCITY_LIMIT,
}


g1_29dof_torsobase_delayed_actuator_cfgs: tuple[ActuatorCfg, ...] = (
  _delayed_pos_cfg(
    _builtin_pos_cfg(
      (".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint"),
      effort_limit=88.0,
      stiffness=90.0,
      damping=2.0,
      armature=0.03,
    ),
    min_delay=0,
    max_delay=1,
  ),
  _delayed_pos_cfg(
    _builtin_pos_cfg(
      (".*_knee_joint",),
      effort_limit=139.0,
      stiffness=140.0,
      damping=2.5,
      armature=0.03,
    ),
    min_delay=0,
    max_delay=1,
  ),
  _delayed_pos_cfg(
    _builtin_pos_cfg(
      ("waist_roll_joint", "waist_pitch_joint"),
      effort_limit=50.0,
      stiffness=60.0,
      damping=2.5,
      armature=0.03,
    ),
    min_delay=0,
    max_delay=1,
  ),
  _delayed_pos_cfg(
    _builtin_pos_cfg(
      ("waist_yaw_joint",),
      effort_limit=88.0,
      stiffness=90.0,
      damping=2.5,
      armature=0.03,
    ),
    min_delay=0,
    max_delay=1,
  ),
  _delayed_pos_cfg(
    _builtin_pos_cfg(
      (".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
      effort_limit=20.0,
      stiffness=20.0,
      damping=1.0,
      armature=0.03,
    ),
    min_delay=0,
    max_delay=1,
  ),
  _delayed_pos_cfg(
    _builtin_pos_cfg(
      (
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
      ),
      effort_limit=25.0,
      stiffness=25.0,
      damping=1.0,
      armature=0.03,
    ),
    min_delay=0,
    max_delay=1,
  ),
  _delayed_pos_cfg(
    _builtin_pos_cfg(
      (".*wrist_roll_joint",),
      effort_limit=25.0,
      stiffness=25.0,
      damping=1.0,
      armature=0.03,
    ),
    min_delay=0,
    max_delay=1,
  ),
  _delayed_pos_cfg(
    _builtin_pos_cfg(
      (".*wrist_pitch_joint", ".*wrist_yaw_joint"),
      effort_limit=5.0,
      stiffness=5.0,
      damping=0.5,
      armature=0.03,
    ),
    min_delay=0,
    max_delay=1,
  ),
)


beyondmimic_g1_29dof_actuator_cfgs: tuple[ActuatorCfg, ...] = (
  _builtin_pos_cfg(
    (".*_hip_pitch_joint", ".*_hip_yaw_joint"),
    effort_limit=88.0,
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    armature=ARMATURE_7520_14,
  ),
  _builtin_pos_cfg(
    ("waist_yaw_joint",),
    effort_limit=88.0,
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    armature=ARMATURE_7520_14,
  ),
  _builtin_pos_cfg(
    (".*_hip_roll_joint", ".*_knee_joint"),
    effort_limit=139.0,
    stiffness=STIFFNESS_7520_22,
    damping=DAMPING_7520_22,
    armature=ARMATURE_7520_22,
  ),
  _builtin_pos_cfg(
    (".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    effort_limit=50.0,
    stiffness=2.0 * STIFFNESS_5020,
    damping=2.0 * DAMPING_5020,
    armature=2.0 * ARMATURE_5020,
  ),
  _builtin_pos_cfg(
    ("waist_roll_joint", "waist_pitch_joint"),
    effort_limit=50.0,
    stiffness=2.0 * STIFFNESS_5020,
    damping=2.0 * DAMPING_5020,
    armature=2.0 * ARMATURE_5020,
  ),
  _builtin_pos_cfg(
    (
      ".*_shoulder_pitch_joint",
      ".*_shoulder_roll_joint",
      ".*_shoulder_yaw_joint",
      ".*_elbow_joint",
      ".*_wrist_roll_joint",
    ),
    effort_limit=25.0,
    stiffness=STIFFNESS_5020,
    damping=DAMPING_5020,
    armature=ARMATURE_5020,
  ),
  _builtin_pos_cfg(
    (".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
    effort_limit=5.0,
    stiffness=STIFFNESS_4010,
    damping=DAMPING_4010,
    armature=ARMATURE_4010,
  ),
)


beyondmimic_g1_29dof_delayed_actuator_cfgs: tuple[ActuatorCfg, ...] = tuple(
  _delayed_pos_cfg(act_cfg, min_delay=0, max_delay=2)
  for act_cfg in beyondmimic_g1_29dof_actuator_cfgs
)


G1_29DOF_TORSOBASE_CFG = _set_robot_actuators(
  _new_g1_cfg(init_state=_SIMPLIFIED_INIT_STATE),
  g1_29dof_torsobase_delayed_actuator_cfgs,
  soft_joint_pos_limit_factor=0.95,
)
G1_29DOF_TORSOBASE_CLOG_CFG = _set_robot_actuators(
  _new_g1_cfg(init_state=_SIMPLIFIED_INIT_STATE),
  g1_29dof_torsobase_delayed_actuator_cfgs,
  soft_joint_pos_limit_factor=0.95,
)
G1_29DOF_TORSOBASE_POPSICLE_CFG = _set_robot_actuators(
  _new_g1_cfg(init_state=_POPSICLE_INIT_STATE),
  beyondmimic_g1_29dof_actuator_cfgs,
  soft_joint_pos_limit_factor=0.9,
)


G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping = [
  1,
  0,
  2,  # waist pitch
  4,
  3,
  5,  # waist roll
  7,
  6,
  8,  # waist yaw
  10,
  9,
  12,
  11,
  14,
  13,
  16,
  15,
  18,
  17,
  20,
  19,
  22,
  21,
  24,
  23,
  26,
  25,
  28,
  27,
]

G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf = [
  1,
  1,
  1,  # waist pitch
  -1,
  -1,  # shoulder roll
  -1,  # waist roll
  -1,
  -1,  # shoulder yaw
  -1,  # waist yaw
  1,
  1,
  1,
  1,
  -1,
  -1,  # wrist roll
  -1,
  -1,  # hip roll
  1,
  1,  # wrist pitch
  -1,
  -1,  # hip yaw
  -1,
  -1,  # wrist yaw
  1,
  1,
  1,
  1,
  -1,
  -1,  # ankle roll
]


# Keep InstinctLab-style metadata exports.
beyondmimic_g1_29dof_actuators = {
  "legs": {
    "joint_names_expr": (
      ".*_hip_yaw_joint",
      ".*_hip_roll_joint",
      ".*_hip_pitch_joint",
      ".*_knee_joint",
    ),
    "effort_limit_sim": {
      ".*_hip_yaw_joint": 88.0,
      ".*_hip_roll_joint": 139.0,
      ".*_hip_pitch_joint": 88.0,
      ".*_knee_joint": 139.0,
    },
    "velocity_limit_sim": {
      ".*_hip_yaw_joint": 32.0,
      ".*_hip_roll_joint": 20.0,
      ".*_hip_pitch_joint": 32.0,
      ".*_knee_joint": 20.0,
    },
    "stiffness": {
      ".*_hip_pitch_joint": STIFFNESS_7520_14,
      ".*_hip_roll_joint": STIFFNESS_7520_22,
      ".*_hip_yaw_joint": STIFFNESS_7520_14,
      ".*_knee_joint": STIFFNESS_7520_22,
    },
    "damping": {
      ".*_hip_pitch_joint": DAMPING_7520_14,
      ".*_hip_roll_joint": DAMPING_7520_22,
      ".*_hip_yaw_joint": DAMPING_7520_14,
      ".*_knee_joint": DAMPING_7520_22,
    },
    "armature": {
      ".*_hip_pitch_joint": ARMATURE_7520_14,
      ".*_hip_roll_joint": ARMATURE_7520_22,
      ".*_hip_yaw_joint": ARMATURE_7520_14,
      ".*_knee_joint": ARMATURE_7520_22,
    },
  },
  "feet": {
    "joint_names_expr": (".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    "effort_limit_sim": 50.0,
    "velocity_limit_sim": 37.0,
    "stiffness": 2.0 * STIFFNESS_5020,
    "damping": 2.0 * DAMPING_5020,
    "armature": 2.0 * ARMATURE_5020,
  },
  "waist": {
    "joint_names_expr": ("waist_roll_joint", "waist_pitch_joint"),
    "effort_limit_sim": 50.0,
    "velocity_limit_sim": 37.0,
    "stiffness": 2.0 * STIFFNESS_5020,
    "damping": 2.0 * DAMPING_5020,
    "armature": 2.0 * ARMATURE_5020,
  },
  "waist_yaw": {
    "joint_names_expr": ("waist_yaw_joint",),
    "effort_limit_sim": 88.0,
    "velocity_limit_sim": 32.0,
    "stiffness": STIFFNESS_7520_14,
    "damping": DAMPING_7520_14,
    "armature": ARMATURE_7520_14,
  },
  "arms": {
    "joint_names_expr": (
      ".*_shoulder_pitch_joint",
      ".*_shoulder_roll_joint",
      ".*_shoulder_yaw_joint",
      ".*_elbow_joint",
      ".*_wrist_roll_joint",
      ".*_wrist_pitch_joint",
      ".*_wrist_yaw_joint",
    ),
    "effort_limit_sim": {
      ".*_shoulder_pitch_joint": 25.0,
      ".*_shoulder_roll_joint": 25.0,
      ".*_shoulder_yaw_joint": 25.0,
      ".*_elbow_joint": 25.0,
      ".*_wrist_roll_joint": 25.0,
      ".*_wrist_pitch_joint": 5.0,
      ".*_wrist_yaw_joint": 5.0,
    },
    "velocity_limit_sim": {
      ".*_shoulder_pitch_joint": 37.0,
      ".*_shoulder_roll_joint": 37.0,
      ".*_shoulder_yaw_joint": 37.0,
      ".*_elbow_joint": 37.0,
      ".*_wrist_roll_joint": 37.0,
      ".*_wrist_pitch_joint": 22.0,
      ".*_wrist_yaw_joint": 22.0,
    },
    "stiffness": {
      ".*_shoulder_pitch_joint": STIFFNESS_5020,
      ".*_shoulder_roll_joint": STIFFNESS_5020,
      ".*_shoulder_yaw_joint": STIFFNESS_5020,
      ".*_elbow_joint": STIFFNESS_5020,
      ".*_wrist_roll_joint": STIFFNESS_5020,
      ".*_wrist_pitch_joint": STIFFNESS_4010,
      ".*_wrist_yaw_joint": STIFFNESS_4010,
    },
    "damping": {
      ".*_shoulder_pitch_joint": DAMPING_5020,
      ".*_shoulder_roll_joint": DAMPING_5020,
      ".*_shoulder_yaw_joint": DAMPING_5020,
      ".*_elbow_joint": DAMPING_5020,
      ".*_wrist_roll_joint": DAMPING_5020,
      ".*_wrist_pitch_joint": DAMPING_4010,
      ".*_wrist_yaw_joint": DAMPING_4010,
    },
    "armature": {
      ".*_shoulder_pitch_joint": ARMATURE_5020,
      ".*_shoulder_roll_joint": ARMATURE_5020,
      ".*_shoulder_yaw_joint": ARMATURE_5020,
      ".*_elbow_joint": ARMATURE_5020,
      ".*_wrist_roll_joint": ARMATURE_5020,
      ".*_wrist_pitch_joint": ARMATURE_4010,
      ".*_wrist_yaw_joint": ARMATURE_4010,
    },
  },
}

beyondmimic_g1_29dof_delayed_actuators = copy.deepcopy(beyondmimic_g1_29dof_actuators)
for actuator_cfg in beyondmimic_g1_29dof_delayed_actuators.values():
  actuator_cfg["min_delay"] = 0
  actuator_cfg["max_delay"] = 2


beyondmimic_action_scale: dict[str, float] = {}
for actuator_cfg in beyondmimic_g1_29dof_actuators.values():
  effort_cfg = actuator_cfg["effort_limit_sim"]
  stiffness_cfg = actuator_cfg["stiffness"]
  for joint_expr in actuator_cfg["joint_names_expr"]:
    effort = _resolve_limit_value(effort_cfg, joint_expr)
    stiffness = _resolve_limit_value(stiffness_cfg, joint_expr)
    if stiffness != 0.0:
      beyondmimic_action_scale[joint_expr] = 0.25 * effort / stiffness


__all__ = [
  "ACTUATOR_4010_EFFORT_LIMIT",
  "ACTUATOR_4010_VELOCITY_LIMIT",
  "ACTUATOR_5020_EFFORT_LIMIT",
  "ACTUATOR_5020_VELOCITY_LIMIT",
  "ACTUATOR_7520_14_EFFORT_LIMIT",
  "ACTUATOR_7520_14_VELOCITY_LIMIT",
  "ACTUATOR_7520_22_EFFORT_LIMIT",
  "ACTUATOR_7520_22_VELOCITY_LIMIT",
  "ACTUATOR_DUAL_5020_EFFORT_LIMIT",
  "ARMATURE_4010",
  "ARMATURE_5020",
  "ARMATURE_7520_14",
  "ARMATURE_7520_22",
  "DAMPING_4010",
  "DAMPING_5020",
  "DAMPING_7520_14",
  "DAMPING_7520_22",
  "DAMPING_RATIO",
  "GEARS_4010",
  "GEARS_5020",
  "GEARS_7520_14",
  "GEARS_7520_22",
  "G1_JOINT_VEL_LIMIT_BY_EFFORT_LIMIT",
  "G1_29DOF_TORSOBASE_CFG",
  "G1_29DOF_TORSOBASE_CLOG_CFG",
  "G1_29DOF_INSTINCTLAB_JOINT_ORDER",
  "G1_29DOF_TORSOBASE_POPSICLE_CFG",
  "G1_MESHES_DIR",
  "G1_MJCF_PATH",
  "G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping",
  "G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf",
  "NATURAL_FREQ",
  "ROTOR_INERTIAS_4010",
  "ROTOR_INERTIAS_5020",
  "ROTOR_INERTIAS_7520_14",
  "ROTOR_INERTIAS_7520_22",
  "STIFFNESS_4010",
  "STIFFNESS_5020",
  "STIFFNESS_7520_14",
  "STIFFNESS_7520_22",
  "beyondmimic_action_scale",
  "beyondmimic_g1_29dof_actuator_cfgs",
  "beyondmimic_g1_29dof_actuators",
  "beyondmimic_g1_29dof_delayed_actuator_cfgs",
  "beyondmimic_g1_29dof_delayed_actuators",
  "get_g1_assets",
  "g1_29dof_torsobase_delayed_actuator_cfgs",
]
