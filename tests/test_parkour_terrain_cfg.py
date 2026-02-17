from instinct_mjlab.tasks.parkour.config.g1.env_cfgs import (
  instinct_g1_parkour_amp_env_cfg,
)


def test_parkour_env_terrain_uses_generator():
  cfg = instinct_g1_parkour_amp_env_cfg()
  assert cfg.scene.terrain.terrain_type == "generator"
  assert cfg.scene.terrain.terrain_generator is not None
  assert cfg.scene.terrain.max_init_terrain_level == 5
  assert "edges" in cfg.scene.terrain.virtual_obstacles
