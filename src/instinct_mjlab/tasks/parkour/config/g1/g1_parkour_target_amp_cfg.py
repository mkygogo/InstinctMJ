"""G1 parkour AMP task config factories.

Mirrors the original InstinctLab ``g1_parkour_target_amp_cfg.py`` using
mjlab-native factory functions.  Config classes (``G1ParkourRoughEnvCfg``,
``G1ParkourEnvCfg``, etc.) are replaced by a single factory
``instinct_g1_parkour_amp_final_cfg(play, shoe)`` that returns a fully-built
``ManagerBasedRlEnvCfg``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import mujoco
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.viewer import ViewerConfig

from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

from instinct_mjlab.assets.unitree_g1 import (
  beyondmimic_action_scale,
  beyondmimic_g1_29dof_delayed_actuator_cfgs,
)
from instinct_mjlab.motion_reference.utils import motion_interpolate_bilinear
from instinct_mjlab.tasks.parkour.config.parkour_env_cfg import (
  set_parkour_amp_observations,
  set_parkour_basic_settings,
  set_parkour_commands,
  set_parkour_curriculum,
  set_parkour_events,
  set_parkour_observations,
  set_parkour_play_overrides,
  set_parkour_rewards,
  set_parkour_scene_sensors,
  set_parkour_terminations,
  set_parkour_terrain,
)
from instinct_mjlab.utils.datasets import resolve_datasets_root

_DATASETS_ROOT = resolve_datasets_root()
_PARKOUR_TASK_DIR = Path(__file__).resolve().parents[2]
_PARKOUR_G1_WITH_SHOE_MJCF_PATH = (
  _PARKOUR_TASK_DIR / "mjcf" / "g1_29dof_torsoBase_popsicle_with_shoe.xml"
)


# ---------------------------------------------------------------------------
# Motion reference configs (mirrors InstinctLab AmassMotionCfg)
# ---------------------------------------------------------------------------


class AmassMotionCfgBase:
  """AmassMotion baseline config used by motion_reference."""
  path = str(_DATASETS_ROOT)
  retargetting_func = None
  filtered_motion_selection_filepath = str(
    _DATASETS_ROOT / "parkour_motion_without_run.yaml"
  )
  motion_start_from_middle_range = [0.0, 0.9]
  motion_start_height_offset = 0.0
  ensure_link_below_zero_ground = False
  buffer_device = "output_device"
  motion_interpolate_func = motion_interpolate_bilinear
  velocity_estimation_method = "frontward"


class AmassMotionCfg(AmassMotionCfgBase):
  pass


# ---------------------------------------------------------------------------
# Shoe spec factory
# ---------------------------------------------------------------------------


def _parkour_g1_with_shoe_spec() -> mujoco.MjSpec:
  """Build MjSpec for the G1 robot with shoe mesh."""
  spec = mujoco.MjSpec.from_file(str(_PARKOUR_G1_WITH_SHOE_MJCF_PATH))
  spec.assets = g1_constants.get_assets(spec.meshdir)
  return spec


def _apply_shoe_config(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply shoe-specific adjustments to a parkour env cfg (in-place).

  Mirrors the ``ShoeConfigMixin.apply_shoe_config()`` from the original
  InstinctLab ``g1_parkour_target_amp_cfg.py``.
  """
  # Replace robot spec with shoe variant
  robot_cfg_with_shoe = copy.deepcopy(cfg.scene.entities["robot"])
  robot_cfg_with_shoe.spec_fn = _parkour_g1_with_shoe_spec
  cfg.scene.entities["robot"] = robot_cfg_with_shoe

  # Adjust leg volume points z-range for shoes
  leg_volume_points = next(
    sensor_cfg for sensor_cfg in cfg.scene.sensors if sensor_cfg.name == "leg_volume_points"
  )
  leg_volume_points.points_generator.z_min = -0.063
  leg_volume_points.points_generator.z_max = -0.023

  # Adjust feet_at_plane height offset for shoes
  cfg.rewards["feet_at_plane"].params["height_offset"] = 0.058


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply play-mode-specific overrides to a parkour env cfg (in-place).

  Mirrors ``G1ParkourRoughEnvCfg_PLAY.__post_init__`` from the original
  InstinctLab ``g1_parkour_target_amp_cfg.py``.
  """
  # Viewer
  cfg.viewer = ViewerConfig(
    lookat=(0.0, 0.75, 0.0),
    distance=4.123105625617661,
    elevation=-14.036243467926479,
    azimuth=180.0,
    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    entity_name="robot",
  )


# ---------------------------------------------------------------------------
# G1-specific actuator setup (from original env_cfgs.py)
# ---------------------------------------------------------------------------


def _set_parkour_actuators(cfg: ManagerBasedRlEnvCfg) -> None:
  """Set G1-specific actuators and action scale for parkour (in-place).

  Mirrors the original InstinctLab ``G1ParkourRoughEnvCfg.__post_init__``
  where ``beyondmimic_g1_29dof_delayed_actuators`` and
  ``beyondmimic_action_scale`` are applied.
  """
  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None
  robot_cfg.articulation.actuators = copy.deepcopy(
    beyondmimic_g1_29dof_delayed_actuator_cfgs
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = copy.deepcopy(beyondmimic_action_scale)


# ---------------------------------------------------------------------------
# Base parkour env builder (merges original env_cfgs.py logic)
# ---------------------------------------------------------------------------


def instinct_g1_parkour_amp_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the base G1 parkour AMP environment configuration.

  Mirrors the original InstinctLab ``G1ParkourRoughEnvCfg`` assembly:
  starts from the tracking base, then applies parkour-specific MDP
  settings (terrain, sensors, observations, rewards, etc.) and
  G1-specific actuators.

  Args:
    play: If True, apply play-mode overrides (fewer envs, relaxed
      termination, etc.).

  Returns:
    A ``ManagerBasedRlEnvCfg`` instance with parkour settings applied.
  """
  # Scene settings (start from tracking base with G1 robot)
  cfg = unitree_g1_flat_tracking_env_cfg(play=play, has_state_estimation=True)

  # Basic settings
  set_parkour_basic_settings(cfg)
  # G1-specific actuators
  _set_parkour_actuators(cfg)
  # Terrain
  set_parkour_terrain(cfg, play=play)
  # Scene sensors
  set_parkour_scene_sensors(cfg)

  # MDP settings
  set_parkour_commands(cfg)
  set_parkour_observations(cfg)
  set_parkour_amp_observations(cfg)
  set_parkour_rewards(cfg)
  set_parkour_curriculum(cfg)
  set_parkour_terminations(cfg)
  set_parkour_events(cfg)

  # general settings
  # simulation settings
  # update sensor update periods
  # lights
  if play:
    set_parkour_play_overrides(cfg)
  return cfg


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def instinct_g1_parkour_amp_final_cfg(
  *,
  play: bool = False,
  shoe: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create the final G1 parkour AMP env configuration.

  Args:
    play: If True, apply play-mode overrides (fewer envs, relaxed
      termination, etc.).
    shoe: If True, apply shoe-specific adjustments (default is True,
      matching the original ``G1ParkourEnvCfg``).

  Returns:
    A fully-built ``ManagerBasedRlEnvCfg`` instance.
  """
  # Build base parkour config (already includes play overrides if requested)
  cfg = instinct_g1_parkour_amp_env_cfg(play=play)

  # Apply shoe-specific adjustments (matches original G1ParkourEnvCfg)
  if shoe:
    _apply_shoe_config(cfg)

  # Apply play-mode viewer overrides
  if play:
    _apply_play_overrides(cfg)

  return cfg


# ---------------------------------------------------------------------------
# Backward-compatible class aliases (thin wrappers for registration)
# ---------------------------------------------------------------------------


class G1ParkourEnvCfg(ManagerBasedRlEnvCfg):
  """G1 parkour train config (with shoe)."""

  def __init__(self):
    cfg = instinct_g1_parkour_amp_final_cfg(play=False, shoe=True)
    super().__init__(**{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()})


class G1ParkourEnvCfg_PLAY(ManagerBasedRlEnvCfg):
  """G1 parkour play config (with shoe)."""

  def __init__(self):
    cfg = instinct_g1_parkour_amp_final_cfg(play=True, shoe=True)
    super().__init__(**{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()})

