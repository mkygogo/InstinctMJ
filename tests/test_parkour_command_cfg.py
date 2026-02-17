from instinct_mjlab.tasks.parkour.config.g1.env_cfgs import (
  _BASE_VELOCITY_COMMAND_NAME,
  _MOTION_COMMAND_NAME,
  instinct_g1_parkour_amp_env_cfg,
)
from instinct_mjlab.tasks.parkour.mdp.commands import PoseVelocityCommandCfg


def test_parkour_env_uses_pose_velocity_command():
  cfg = instinct_g1_parkour_amp_env_cfg()

  assert _BASE_VELOCITY_COMMAND_NAME in cfg.commands
  assert _MOTION_COMMAND_NAME in cfg.commands
  assert isinstance(cfg.commands[_BASE_VELOCITY_COMMAND_NAME], PoseVelocityCommandCfg)


def test_parkour_velocity_ranges_follow_terrain_names():
  cfg = instinct_g1_parkour_amp_env_cfg()
  command_cfg = cfg.commands[_BASE_VELOCITY_COMMAND_NAME]
  assert isinstance(command_cfg, PoseVelocityCommandCfg)

  terrain_names = set(cfg.scene.terrain.terrain_generator.sub_terrains.keys())
  velocity_range_names = set(command_cfg.velocity_ranges.keys())
  random_velocity_names = set(command_cfg.random_velocity_terrain)

  assert velocity_range_names <= terrain_names
  assert random_velocity_names <= terrain_names


def test_parkour_reward_and_event_use_base_velocity_command():
  cfg = instinct_g1_parkour_amp_env_cfg()

  assert (
    cfg.rewards["track_lin_vel_xy_exp"].params["command_name"]
    == _BASE_VELOCITY_COMMAND_NAME
  )
  assert cfg.rewards["dont_wait"].params["command_name"] == _BASE_VELOCITY_COMMAND_NAME
  assert cfg.events["push_robot"].params["command_name"] == _BASE_VELOCITY_COMMAND_NAME


def test_parkour_amp_reference_still_uses_motion_command():
  cfg = instinct_g1_parkour_amp_env_cfg()
  assert cfg.observations["amp_reference"].terms["base_lin_vel"].params["command_name"] == _MOTION_COMMAND_NAME
  assert cfg.observations["amp_reference"].terms["joint_pos_rel"].params["command_name"] == _MOTION_COMMAND_NAME


def test_parkour_actor_critic_command_observations_use_base_velocity():
  cfg = instinct_g1_parkour_amp_env_cfg()
  assert cfg.observations["actor"].terms["command"].params["command_name"] == _BASE_VELOCITY_COMMAND_NAME
  assert cfg.observations["critic"].terms["command"].params["command_name"] == _BASE_VELOCITY_COMMAND_NAME
