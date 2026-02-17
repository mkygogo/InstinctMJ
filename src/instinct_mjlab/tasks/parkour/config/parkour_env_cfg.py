"""mjlab-native Parkour environment config.

This module defines the base parkour environment configuration used across all
parkour task variants.  Mirrors the original InstinctLab ``parkour_env_cfg.py``
including terrain definitions, scene sensors, observations, rewards, commands,
terminations, events, and curriculum configurations.

All manager configs are defined as factory functions returning
``dict[str, XxxTermCfg]``, following the Instinct_mjlab convention.
"""

from __future__ import annotations

import copy
import math

import mjlab.envs.mdp as envs_mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastSensorCfg,
)
from mjlab.terrains import FlatPatchSamplingCfg
from mjlab.utils.noise import UniformNoiseCfg

import instinct_mjlab.envs.mdp as instinct_envs_mdp
from instinct_mjlab.sensors.volume_points import (
  Grid3dPointsGeneratorCfg,
  VolumePointsCfg,
)
from instinct_mjlab.tasks.mdp import (
  PerceptiveRaycastNoisedHistory,
  parkour_amp_reference_base_ang_vel,
  parkour_amp_reference_base_lin_vel,
  parkour_amp_reference_joint_pos_rel,
  parkour_amp_reference_joint_vel_rel,
  parkour_amp_reference_projected_gravity,
)
import instinct_mjlab.tasks.parkour.mdp as parkour_mdp
from instinct_mjlab.tasks.parkour.mdp.commands import PoseVelocityCommandCfg
from instinct_mjlab.terrains.height_field.hf_terrains_cfg import (
  PerlinDiscreteObstaclesTerrainCfg,
  PerlinInvertedPyramidSlopedTerrainCfg,
  PerlinInvertedPyramidStairsTerrainCfg,
  PerlinPlaneTerrainCfg,
  PerlinPyramidStairsTerrainCfg,
  PerlinSquareGapTerrainCfg,
)
from instinct_mjlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinct_mjlab.terrains.terrain_importer_cfg import TerrainImporterCfg
from instinct_mjlab.terrains.trimesh.mesh_terrains_cfg import (
  PerlinMeshRandomMultiBoxTerrainCfg,
)
from instinct_mjlab.terrains.virtual_obstacle.edge_cylinder_cfg import (
  GreedyconcatEdgeCylinderCfg,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PARKOUR_AMP_HISTORY_LENGTH = 10
_PARKOUR_PROPRIO_HISTORY_LENGTH = 8
_PARKOUR_DEPTH_HISTORY_LENGTH = 37
_PARKOUR_DEPTH_SKIP_FRAMES = 5
_MOTION_COMMAND_NAME = "motion"
_BASE_VELOCITY_COMMAND_NAME = "base_velocity"
_FEET_CONTACT_SENSOR_NAME = "contact_forces"
_TORSO_CONTACT_SENSOR_NAME = "torso_contact_forces"
_UNDESIRED_CONTACT_SENSOR_NAME = "undesired_contact_forces"
_LEG_VOLUME_POINTS_SENSOR_NAME = "leg_volume_points"
_LEFT_HEIGHT_SCANNER_NAME = "left_height_scanner"
_RIGHT_HEIGHT_SCANNER_NAME = "right_height_scanner"
_DEPTH_CAMERA_SENSOR_NAME = "camera"
_PARKOUR_BASE_VELOCITY_RANGES = {
  "perlin_rough": {
    "lin_vel_x": (0.45, 1.0),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (-1.0, 1.0),
  },
  "perlin_rough_stand": {
    "lin_vel_x": (0.0, 0.0),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (0.0, 0.0),
  },
  "square_gaps": {
    "lin_vel_x": (0.45, 0.8),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (-1.0, 1.0),
  },
  "pyramid_stairs": {
    "lin_vel_x": (0.45, 0.8),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (-1.0, 1.0),
  },
  "pyramid_stairs_high": {
    "lin_vel_x": (0.45, 0.8),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (-1.0, 1.0),
  },
  "pyramid_stairs_inv": {
    "lin_vel_x": (0.45, 0.8),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (-1.0, 1.0),
  },
  "pyramid_stairs_inv_high": {
    "lin_vel_x": (0.45, 0.8),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (-1.0, 1.0),
  },
  "boxes": {
    "lin_vel_x": (0.45, 0.8),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (-1.0, 1.0),
  },
  "mesh_boxes": {
    "lin_vel_x": (0.45, 0.8),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (-1.0, 1.0),
  },
  "hf_pyramid_slope_inv": {
    "lin_vel_x": (0.45, 0.8),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (-1.0, 1.0),
  },
}


##
# Scene definition -- Terrain
##

ROUGH_TERRAINS_CFG = FiledTerrainGeneratorCfg(
  seed=0,
  size=(8.0, 8.0),
  border_width=3,
  num_rows=10,
  num_cols=20,
  horizontal_scale=0.05,
  vertical_scale=0.005,
  slope_threshold=1.0,
  use_cache=False,
  curriculum=True,
  sub_terrains={
    "perlin_rough": PerlinPlaneTerrainCfg(
      proportion=0.05,
      noise_scale=[0.0, 0.1],
      noise_frequency=20,
      fractal_octaves=2,
      fractal_lacunarity=2.0,
      fractal_gain=0.25,
      centering=True,
      wall_prob=[0.3, 0.3, 0.3, 0.3],
      wall_height=5.0,
      wall_thickness=0.05,
      flat_patch_sampling={
        "target": FlatPatchSamplingCfg(
          num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
        ),
      },
    ),
    "perlin_rough_stand": PerlinPlaneTerrainCfg(
      proportion=0.05,
      noise_scale=[0.0, 0.1],
      noise_frequency=20,
      fractal_octaves=2,
      fractal_lacunarity=2.0,
      fractal_gain=0.25,
      centering=True,
      wall_prob=[0.3, 0.3, 0.3, 0.3],
      wall_height=5.0,
      wall_thickness=0.05,
      flat_patch_sampling={
        "target": FlatPatchSamplingCfg(
          num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
        ),
      },
    ),
    "square_gaps": PerlinSquareGapTerrainCfg(
      proportion=0.10,
      gap_distance_range=(0.1, 0.7),
      gap_depth=(0.4, 0.6),
      platform_width=2.5,
      border_width=1.0,
      wall_prob=[0.3, 0.3, 0.3, 0.3],
      wall_height=5.0,
      wall_thickness=0.05,
      flat_patch_sampling={
        "target": FlatPatchSamplingCfg(
          num_patches=50,
          patch_radius=[0.05, 0.10, 0.15, 0.20],
          max_height_diff=0.05,
          x_range=(3.7, 3.7),
          y_range=(-0.0, 0.0),
        ),
      },
    ),
    "pyramid_stairs": PerlinPyramidStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.05, 0.23),
      step_width=0.3,
      platform_width=2.5,
      border_width=1.0,
      wall_prob=[0.3, 0.3, 0.3, 0.3],
      wall_height=5.0,
      wall_thickness=0.05,
      perlin_cfg=PerlinPlaneTerrainCfg(
        noise_scale=0.05,
        noise_frequency=20,
        fractal_octaves=2,
        fractal_lacunarity=2.0,
        fractal_gain=0.25,
        centering=True,
      ),
      flat_patch_sampling={
        "target": FlatPatchSamplingCfg(
          num_patches=50,
          patch_radius=[0.05, 0.10, 0.15, 0.20],
          max_height_diff=0.05,
          x_range=(3.7, 3.7),
          y_range=(-0.0, 0.0),
        ),
      },
    ),
    "pyramid_stairs_high": PerlinPyramidStairsTerrainCfg(
      proportion=0.10,
      step_height_range=(0.05, 0.45),
      step_width=1.5,
      platform_width=4.0,
      border_width=1.0,
      wall_prob=[0.3, 0.3, 0.3, 0.3],
      wall_height=5.0,
      wall_thickness=0.05,
      perlin_cfg=PerlinPlaneTerrainCfg(
        noise_scale=0.05,
        noise_frequency=20,
        fractal_octaves=2,
        fractal_lacunarity=2.0,
        fractal_gain=0.25,
        centering=True,
      ),
      flat_patch_sampling={
        "target": FlatPatchSamplingCfg(
          num_patches=50,
          patch_radius=[0.05, 0.10, 0.15, 0.20],
          max_height_diff=0.05,
          x_range=(3.7, 3.7),
          y_range=(-0.0, 0.0),
        ),
      },
    ),
    "pyramid_stairs_inv": PerlinInvertedPyramidStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.05, 0.23),
      step_width=0.3,
      platform_width=2.5,
      border_width=1.0,
      wall_prob=[0.3, 0.3, 0.3, 0.3],
      wall_height=5.0,
      wall_thickness=0.05,
      perlin_cfg=PerlinPlaneTerrainCfg(
        noise_scale=0.05,
        noise_frequency=20,
        fractal_octaves=2,
        fractal_lacunarity=2.0,
        fractal_gain=0.25,
        centering=True,
      ),
      flat_patch_sampling={
        "target": FlatPatchSamplingCfg(
          num_patches=50,
          patch_radius=[0.05, 0.10, 0.15, 0.20],
          max_height_diff=0.05,
          x_range=(3.7, 3.7),
          y_range=(-0.0, 0.0),
        ),
      },
    ),
    "pyramid_stairs_inv_high": PerlinInvertedPyramidStairsTerrainCfg(
      proportion=0.10,
      step_height_range=(0.05, 0.45),
      step_width=1.5,
      platform_width=4.0,
      border_width=1.0,
      wall_prob=[0.3, 0.3, 0.3, 0.3],
      wall_height=5.0,
      wall_thickness=0.05,
      perlin_cfg=PerlinPlaneTerrainCfg(
        noise_scale=0.05,
        noise_frequency=20,
        fractal_octaves=2,
        fractal_lacunarity=2.0,
        fractal_gain=0.25,
        centering=True,
      ),
      flat_patch_sampling={
        "target": FlatPatchSamplingCfg(
          num_patches=50,
          patch_radius=[0.05, 0.10, 0.15, 0.20],
          max_height_diff=0.05,
          x_range=(3.7, 3.7),
          y_range=(-0.0, 0.0),
        ),
      },
    ),
    "boxes": PerlinDiscreteObstaclesTerrainCfg(
      proportion=0.10,
      num_obstacles=20,
      obstacle_height_mode="fixed",
      obstacle_width_range=(0.8, 1.5),
      obstacle_height_range=(0.05, 0.45),
      platform_width=1.5,
      border_width=0.0,
      wall_prob=[0.3, 0.3, 0.3, 0.3],
      wall_height=5.0,
      wall_thickness=0.05,
      perlin_cfg=PerlinPlaneTerrainCfg(
        noise_scale=0.05,
        noise_frequency=20,
        fractal_octaves=2,
        fractal_lacunarity=2.0,
        fractal_gain=0.25,
        centering=True,
      ),
      flat_patch_sampling={
        "target": FlatPatchSamplingCfg(
          num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
        ),
      },
    ),
    "mesh_boxes": PerlinMeshRandomMultiBoxTerrainCfg(
      proportion=0.10,
      box_height_mean=[0.1, 0.4],
      box_height_range=0.05,
      box_length_mean=0.4,
      box_length_range=0.1,
      box_width_mean=0.4,
      box_width_range=0.1,
      platform_width=1.5,
      generation_ratio=0.3,
      no_perlin_at_obstacle=True,
      wall_prob=[0.3, 0.3, 0.3, 0.3],
      wall_height=5.0,
      wall_thickness=0.05,
      flat_patch_sampling={
        "target": FlatPatchSamplingCfg(
          num_patches=50, patch_radius=[0.05, 0.10, 0.15], max_height_diff=0.05
        ),
      },
    ),
    "hf_pyramid_slope_inv": PerlinInvertedPyramidSlopedTerrainCfg(
      proportion=0.10,
      slope_range=(0.0, 0.7),
      platform_width=1.5,
      border_width=1.0,
      wall_prob=[0.3, 0.3, 0.3, 0.3],
      wall_height=5.0,
      wall_thickness=0.05,
      perlin_cfg=PerlinPlaneTerrainCfg(
        noise_scale=0.00,
        noise_frequency=20,
        fractal_octaves=2,
        fractal_lacunarity=2.0,
        fractal_gain=0.25,
        centering=True,
      ),
      flat_patch_sampling={
        "target": FlatPatchSamplingCfg(
          num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
        ),
      },
    ),
  },
)

ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for _sub_terrain_name, _sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
  _sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]
ROUGH_TERRAINS_CFG_PLAY.num_rows = 4
ROUGH_TERRAINS_CFG_PLAY.num_cols = 10


def rough_terrains_cfg(play: bool = False) -> FiledTerrainGeneratorCfg:
  """Return a deep copy of the rough terrain config for train or play mode."""
  return copy.deepcopy(ROUGH_TERRAINS_CFG_PLAY if play else ROUGH_TERRAINS_CFG)


# ---------------------------------------------------------------------------
# Scene definition -- Sensors
# ---------------------------------------------------------------------------


def set_parkour_scene_sensors(cfg: ManagerBasedRlEnvCfg) -> None:
  """Set parkour-specific sensors on the scene (in-place).

  Mirrors the original InstinctLab ``SceneCfg`` sensor definitions:
  contact_forces, torso_contact_forces, undesired_contact_forces,
  leg_volume_points, left/right_height_scanner, and camera.
  """
  replace_names = {
    _FEET_CONTACT_SENSOR_NAME,
    _TORSO_CONTACT_SENSOR_NAME,
    _UNDESIRED_CONTACT_SENSOR_NAME,
    _LEG_VOLUME_POINTS_SENSOR_NAME,
    _LEFT_HEIGHT_SCANNER_NAME,
    _RIGHT_HEIGHT_SCANNER_NAME,
    _DEPTH_CAMERA_SENSOR_NAME,
  }
  existing_sensors = tuple(
    sensor_cfg for sensor_cfg in cfg.scene.sensors if sensor_cfg.name not in replace_names
  )
  feet_contact_sensor = ContactSensorCfg(
    name=_FEET_CONTACT_SENSOR_NAME,
    primary=ContactMatch(
      mode="body",
      pattern=("left_ankle_roll_link", "right_ankle_roll_link"),
      entity="robot",
    ),
    fields=("found", "force"),
    reduce="maxforce",
    track_air_time=True,
    history_length=3,
  )
  torso_contact_sensor = ContactSensorCfg(
    name=_TORSO_CONTACT_SENSOR_NAME,
    primary=ContactMatch(mode="body", pattern="torso_link", entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    track_air_time=False,
    history_length=3,
  )
  undesired_contact_sensor = ContactSensorCfg(
    name=_UNDESIRED_CONTACT_SENSOR_NAME,
    primary=ContactMatch(
      mode="body",
      pattern=".*",
      entity="robot",
      exclude=("left_ankle_roll_link", "right_ankle_roll_link"),
    ),
    fields=("found", "force"),
    reduce="maxforce",
    track_air_time=False,
    history_length=3,
  )
  leg_volume_points_sensor = VolumePointsCfg(
    name=_LEG_VOLUME_POINTS_SENSOR_NAME,
    entity_name="robot",
    body_names=".*_ankle_roll_link",
    points_generator=Grid3dPointsGeneratorCfg(
      x_min=-0.025,
      x_max=0.12,
      x_num=10,
      y_min=-0.03,
      y_max=0.03,
      y_num=5,
      z_min=-0.04,
      z_max=0.0,
      z_num=2,
    ),
    debug_vis=False,
  )
  left_height_scanner = RayCastSensorCfg(
    name=_LEFT_HEIGHT_SCANNER_NAME,
    frame=ObjRef(type="body", name="left_ankle_roll_link", entity="robot"),
    pattern=GridPatternCfg(resolution=0.12, size=(0.12, 0.0)),
    ray_alignment="yaw",
    max_distance=10.0,
    debug_vis=False,
  )
  right_height_scanner = RayCastSensorCfg(
    name=_RIGHT_HEIGHT_SCANNER_NAME,
    frame=ObjRef(type="body", name="right_ankle_roll_link", entity="robot"),
    pattern=GridPatternCfg(resolution=0.12, size=(0.12, 0.0)),
    ray_alignment="yaw",
    max_distance=10.0,
    debug_vis=False,
  )
  depth_camera_sensor = RayCastSensorCfg(
    name=_DEPTH_CAMERA_SENSOR_NAME,
    frame=ObjRef(type="body", name="torso_link", entity="robot"),
    pattern=PinholeCameraPatternCfg(width=64, height=36, fovy=58.29),
    ray_alignment="base",
    max_distance=2.5,
    debug_vis=False,
  )
  cfg.scene.sensors = existing_sensors + (
    feet_contact_sensor,
    torso_contact_sensor,
    undesired_contact_sensor,
    leg_volume_points_sensor,
    left_height_scanner,
    right_height_scanner,
    depth_camera_sensor,
  )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def set_parkour_commands(cfg: ManagerBasedRlEnvCfg) -> None:
  """Set parkour-specific command configuration (in-place).

  Mirrors the original InstinctLab ``CommandsCfg``.
  """
  ranges_cfg = PoseVelocityCommandCfg.Ranges(
    lin_vel_x=(0.0, 0.0),
    lin_vel_y=(0.0, 0.0),
    ang_vel_z=(-1.0, 1.0),
  )
  cfg.commands[_BASE_VELOCITY_COMMAND_NAME] = PoseVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(8.0, 12.0),
    debug_vis=False,
    velocity_control_stiffness=2.0,
    heading_control_stiffness=2.0,
    rel_standing_envs=0.05,
    ranges=ranges_cfg,
    random_velocity_terrain=["perlin_rough_stand"],
    velocity_ranges=copy.deepcopy(_PARKOUR_BASE_VELOCITY_RANGES),
    only_positive_lin_vel_x=True,
    lin_vel_threshold=0.0,
    ang_vel_threshold=0.0,
    target_dis_threshold=0.4,
  )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def set_parkour_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  """Set parkour-specific actor/critic observation groups (in-place).

  Mirrors the original InstinctLab ``ObservationsCfg.PolicyCfg`` and
  ``ObservationsCfg.CriticCfg``.
  """
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
      history_length=_PARKOUR_PROPRIO_HISTORY_LENGTH,
      flatten_history_dim=True,
      scale=0.25,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
      history_length=_PARKOUR_PROPRIO_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "velocity_commands": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": _BASE_VELOCITY_COMMAND_NAME},
      history_length=_PARKOUR_PROPRIO_HISTORY_LENGTH,
      flatten_history_dim=True,
      noise=None,
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
      history_length=_PARKOUR_PROPRIO_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
      scale=0.05,
      history_length=_PARKOUR_PROPRIO_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "actions": ObservationTermCfg(
      func=envs_mdp.last_action,
      history_length=_PARKOUR_PROPRIO_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "depth_image": ObservationTermCfg(
      func=PerceptiveRaycastNoisedHistory,
      params={
        # noise
        "sensor_name": _DEPTH_CAMERA_SENSOR_NAME,
        "min_depth": 0.0,
        "max_depth": 2.5,
        "crop_top": 18,
        "crop_bottom": 2,
        "crop_left": 0,
        "crop_right": 48,
        "output_height": 16,
        "output_width": 16,
        "history_length": _PARKOUR_DEPTH_HISTORY_LENGTH,
        "history_skip_frames": _PARKOUR_DEPTH_SKIP_FRAMES,
        "update_period_s": 0.02,
      },
      noise=None,
    ),
  }
  actor_terms["command"] = actor_terms["velocity_commands"]
  critic_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.base_lin_vel,
      history_length=_PARKOUR_PROPRIO_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      history_length=_PARKOUR_PROPRIO_HISTORY_LENGTH,
      flatten_history_dim=True,
      scale=0.25,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      history_length=_PARKOUR_PROPRIO_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "velocity_commands": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": _BASE_VELOCITY_COMMAND_NAME},
      history_length=_PARKOUR_PROPRIO_HISTORY_LENGTH,
      flatten_history_dim=True,
      noise=None,
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      history_length=_PARKOUR_PROPRIO_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      scale=0.05,
      history_length=_PARKOUR_PROPRIO_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "actions": ObservationTermCfg(
      func=envs_mdp.last_action,
      history_length=_PARKOUR_PROPRIO_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "depth_image": ObservationTermCfg(
      func=PerceptiveRaycastNoisedHistory,
      params={
        "sensor_name": _DEPTH_CAMERA_SENSOR_NAME,
        "min_depth": 0.0,
        "max_depth": 2.5,
        "crop_top": 18,
        "crop_bottom": 2,
        "crop_left": 0,
        "crop_right": 48,
        "output_height": 16,
        "output_width": 16,
        "history_length": _PARKOUR_DEPTH_HISTORY_LENGTH,
        "history_skip_frames": _PARKOUR_DEPTH_SKIP_FRAMES,
        "update_period_s": 0.02,
      },
      noise=None,
    ),
  }
  critic_terms["command"] = critic_terms["velocity_commands"]
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=False,
    enable_corruption=True,
    nan_policy="sanitize",
    nan_check_per_term=True,
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=False,
    enable_corruption=False,
    nan_policy="sanitize",
    nan_check_per_term=True,
  )


def set_parkour_amp_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  """Set AMP policy/reference observation groups (in-place).

  Mirrors the original InstinctLab ``ObservationsCfg.AmpPolicyStateObsCfg``
  and ``ObservationsCfg.AmpReferenceStateObsCfg``.
  """
  amp_policy_terms = {
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      history_length=_PARKOUR_AMP_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "joint_pos_rel": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg(name="robot", preserve_order=True)},
      history_length=_PARKOUR_AMP_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg(name="robot", preserve_order=True)},
      scale=0.05,
      history_length=_PARKOUR_AMP_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.base_lin_vel,
      params={"asset_cfg": SceneEntityCfg(name="robot")},
      history_length=_PARKOUR_AMP_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      params={"asset_cfg": SceneEntityCfg(name="robot")},
      history_length=_PARKOUR_AMP_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
  }
  amp_reference_terms = {
    "projected_gravity": ObservationTermCfg(
      func=parkour_amp_reference_projected_gravity,
      params={"command_name": _MOTION_COMMAND_NAME},
      history_length=_PARKOUR_AMP_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "joint_pos_rel": ObservationTermCfg(
      func=parkour_amp_reference_joint_pos_rel,
      params={"command_name": _MOTION_COMMAND_NAME, "robot_name": "robot"},
      history_length=_PARKOUR_AMP_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "joint_vel": ObservationTermCfg(
      func=parkour_amp_reference_joint_vel_rel,
      params={"command_name": _MOTION_COMMAND_NAME, "robot_name": "robot"},
      scale=0.05,
      history_length=_PARKOUR_AMP_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=parkour_amp_reference_base_lin_vel,
      params={"command_name": _MOTION_COMMAND_NAME},
      history_length=_PARKOUR_AMP_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=parkour_amp_reference_base_ang_vel,
      params={"command_name": _MOTION_COMMAND_NAME},
      history_length=_PARKOUR_AMP_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
  }
  cfg.observations["amp_policy"] = ObservationGroupCfg(
    terms=amp_policy_terms,
    concatenate_terms=False,
    enable_corruption=False,
    nan_policy="sanitize",
    nan_check_per_term=True,
  )
  cfg.observations["amp_reference"] = ObservationGroupCfg(
    terms=amp_reference_terms,
    concatenate_terms=False,
    enable_corruption=False,
    nan_policy="sanitize",
    nan_check_per_term=True,
  )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def set_parkour_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Set parkour reward terms (in-place).

  Mirrors the original InstinctLab ``G1Rewards`` (task rewards,
  regularization rewards, and safety rewards).
  """
  cfg.rewards = {
    # ---------- Task rewards ----------
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=parkour_mdp.track_lin_vel_xy_exp,
      weight=2.0,
      params={"command_name": _BASE_VELOCITY_COMMAND_NAME, "std": 0.5},
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=parkour_mdp.track_ang_vel_z_exp,
      weight=2.0,
      params={"command_name": _BASE_VELOCITY_COMMAND_NAME, "std": 0.5},
    ),
    "heading_error": RewardTermCfg(
      func=parkour_mdp.heading_error,
      weight=-1.0,
      params={"command_name": _BASE_VELOCITY_COMMAND_NAME},
    ),
    "dont_wait": RewardTermCfg(
      func=parkour_mdp.dont_wait,
      weight=-0.5,
      params={"command_name": _BASE_VELOCITY_COMMAND_NAME},
    ),
    "is_alive": RewardTermCfg(func=envs_mdp.is_alive, weight=3.0),
    "stand_still": RewardTermCfg(
      func=parkour_mdp.stand_still,
      weight=-0.3,
      params={"command_name": _BASE_VELOCITY_COMMAND_NAME, "offset": 4.0},
    ),
    # ---------- Regularization rewards ----------
    "volume_points_penetration": RewardTermCfg(
      func=parkour_mdp.volume_points_penetration,
      weight=-4.0,
      params={"sensor_name": _LEG_VOLUME_POINTS_SENSOR_NAME},
    ),
    "feet_air_time": RewardTermCfg(
      func=parkour_mdp.feet_air_time,
      weight=0.5,
      params={
        "command_name": _BASE_VELOCITY_COMMAND_NAME,
        "sensor_name": _FEET_CONTACT_SENSOR_NAME,
        "vel_threshold": 0.15,
      },
    ),
    "feet_slide": RewardTermCfg(
      func=parkour_mdp.feet_slide,
      weight=-0.4,
      params={
        "sensor_name": _FEET_CONTACT_SENSOR_NAME,
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        ),
        "threshold": 1.0,
      },
    ),
    "joint_deviation_hip": RewardTermCfg(
      func=parkour_mdp.joint_deviation_square,
      weight=-0.5,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(".*_hip_yaw_joint", ".*_hip_roll_joint"),
        )
      },
    ),
    "ang_vel_xy_l2": RewardTermCfg(func=parkour_mdp.ang_vel_xy_l2, weight=-0.05),
    "dof_torques_l2": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-1.5e-7,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"),
        )
      },
    ),
    "dof_acc_l2": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=-1.25e-7,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "dof_vel_l2": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-1e-4,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.005),
    "flat_orientation_l2": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-3.0),
    "pelvis_orientation_l2": RewardTermCfg(
      func=parkour_mdp.link_orientation,
      weight=-3.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
    ),
    "feet_flat_ori": RewardTermCfg(
      func=parkour_mdp.feet_orientation_contact,
      weight=-0.4,
      params={
        "sensor_name": _FEET_CONTACT_SENSOR_NAME,
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        ),
      },
    ),
    "feet_at_plane": RewardTermCfg(
      func=parkour_mdp.feet_at_plane,
      weight=-0.1,
      params={
        "contact_sensor_name": _FEET_CONTACT_SENSOR_NAME,
        "left_height_scanner_name": _LEFT_HEIGHT_SCANNER_NAME,
        "right_height_scanner_name": _RIGHT_HEIGHT_SCANNER_NAME,
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        ),
        "height_offset": 0.035,
      },
    ),
    "feet_close_xy": RewardTermCfg(
      func=parkour_mdp.feet_close_xy_gauss,
      weight=0.4,
      params={
        "threshold": 0.12,
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        ),
        "std": math.sqrt(0.05),
      },
    ),
    "energy": RewardTermCfg(
      func=parkour_mdp.motors_power_square,
      weight=-5e-5,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"),
        ),
        "normalize_by_stiffness": True,
      },
    ),
    "freeze_upper_body": RewardTermCfg(
      func=parkour_mdp.joint_deviation_l1,
      weight=-0.004,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(".*_shoulder_.*", ".*_elbow_.*", ".*_wrist.*", "waist_.*"),
        )
      },
    ),
    # ---------- Safety rewards ----------
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "dof_vel_limits": RewardTermCfg(
      func=parkour_mdp.joint_vel_limits,
      weight=-1.0,
      params={
        "soft_ratio": 0.9,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "torque_limits": RewardTermCfg(
      func=parkour_mdp.applied_torque_limits_by_ratio,
      weight=-0.01,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "limit_ratio": 0.8,
      },
    ),
    "undesired_contacts": RewardTermCfg(
      func=parkour_mdp.undesired_contacts,
      weight=-1.0,
      params={"sensor_name": _UNDESIRED_CONTACT_SENSOR_NAME, "threshold": 1.0},
    ),
  }


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def set_parkour_terminations(cfg: ManagerBasedRlEnvCfg) -> None:
  """Set parkour termination terms (in-place).

  Mirrors the original InstinctLab ``TerminationsCfg``.
  """
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "terrain_out_bound": TerminationTermCfg(
      func=parkour_mdp.terrain_out_of_bounds,
      time_out=True,
      params={"distance_buffer": 2.0},
    ),
    "base_contact": TerminationTermCfg(
      func=parkour_mdp.illegal_contact,
      params={"sensor_name": _TORSO_CONTACT_SENSOR_NAME, "threshold": 1.0},
    ),
    "bad_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": 1.0},
    ),
    "root_height": TerminationTermCfg(
      func=parkour_mdp.root_height_below_env_origin_minimum,
      params={"minimum_height": 0.5},
    ),
  }


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def set_parkour_events(cfg: ManagerBasedRlEnvCfg) -> None:
  """Set parkour event terms (in-place).

  Mirrors the original InstinctLab ``EventCfg``.
  """
  cfg.events = {
    "physics_material": EventTermCfg(
      func=parkour_mdp.randomize_rigid_body_material,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
        "static_friction_range": (0.3, 1.6),
        "dynamic_friction_range": (0.3, 1.6),
        "restitution_range": (0.05, 0.5),
        "num_buckets": 64,
        "make_consistent": True,
      },
    ),
    # reset
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
        "velocity_range": {
          "x": (-0.2, 0.2),
          "y": (-0.2, 0.2),
          "z": (-0.2, 0.2),
          "roll": (-0.2, 0.2),
          "pitch": (-0.2, 0.2),
          "yaw": (-0.2, 0.2),
        },
      },
    ),
    "register_virtual_obstacles": EventTermCfg(
      func=instinct_envs_mdp.register_virtual_obstacle_to_sensor,
      mode="startup",
      params={"sensor_cfgs": SceneEntityCfg(_LEG_VOLUME_POINTS_SENSOR_NAME)},
    ),
    "push_robot": EventTermCfg(
      func=parkour_mdp.push_by_setting_velocity_without_stand,
      mode="interval",
      interval_range_s=(6.0, 10.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "roll": (0.0, 0.0),
          "pitch": (0.0, 0.0),
          "yaw": (-0.5, 0.5),
        },
        "command_name": _BASE_VELOCITY_COMMAND_NAME,
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.15, 0.15),
        "velocity_range": (0.0, 0.0),
      },
    ),
  }


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


def set_parkour_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  """Set parkour curriculum terms (in-place).

  Mirrors the original InstinctLab ``CurriculumCfg``.
  """
  cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=parkour_mdp.tracking_exp_vel,
      params={
        "lin_vel_threshold": (0.3, 0.6),
        "ang_vel_threshold": (0.0, 0.0),
      },
    ),
  }


# ---------------------------------------------------------------------------
# Terrain setup helper
# ---------------------------------------------------------------------------


def set_parkour_terrain(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
  """Set parkour terrain configuration (in-place)."""
  terrain_gen = rough_terrains_cfg(play=play)
  cfg.scene.terrain = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=copy.deepcopy(terrain_gen),
    max_init_terrain_level=5,
    virtual_obstacles={
      "edges": GreedyconcatEdgeCylinderCfg(
        cylinder_radius=0.05,
        min_points=2,
      ),
    },
  )


# ---------------------------------------------------------------------------
# Basic settings helper
# ---------------------------------------------------------------------------


def set_parkour_basic_settings(cfg: ManagerBasedRlEnvCfg) -> None:
  """Set parkour basic environment settings (in-place).

  Mirrors the original InstinctLab ``ParkourEnvCfg.__post_init__`` settings.
  """
  cfg.scene.num_envs = 4096
  cfg.scene.env_spacing = 2.5
  cfg.episode_length_s = 20.0


# ---------------------------------------------------------------------------
# Play overrides helper
# ---------------------------------------------------------------------------


def set_parkour_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply play-mode overrides to a parkour env cfg (in-place).

  Mirrors the original InstinctLab ``G1ParkourRoughEnvCfg_PLAY.__post_init__``.
  """
  cfg.scene.num_envs = 10
  cfg.scene.env_spacing = 2.5
  cfg.episode_length_s = 10.0

  # spawn the robot randomly in the grid (instead of their terrain levels)
  # reduce the number of terrains to save memory
  if cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.num_rows = 4
    cfg.scene.terrain.terrain_generator.num_cols = 10

  for sensor_cfg in cfg.scene.sensors:
    if sensor_cfg.name == _LEG_VOLUME_POINTS_SENSOR_NAME:
      sensor_cfg.debug_vis = True

  cfg.commands[_BASE_VELOCITY_COMMAND_NAME].debug_vis = True

  cfg.terminations["root_height"] = None
  cfg.events["physics_material"] = None
  cfg.events["push_robot"] = None

  cfg.events["reset_robot_joints"].params = {
    "position_range": (0.0, 0.0),
    "velocity_range": (0.0, 0.0),
  }
