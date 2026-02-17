from instinct_mjlab.tasks.parkour.config.g1.env_cfgs import instinct_g1_parkour_amp_env_cfg


def test_virtual_obstacle_edge_cfg_import():
  from instinct_mjlab.terrains.virtual_obstacle.edge_cylinder_cfg import EdgeCylinderCfg

  assert EdgeCylinderCfg is not None


def test_parkour_terrain_has_virtual_obstacles():
  cfg = instinct_g1_parkour_amp_env_cfg(play=False)
  assert hasattr(cfg.scene.terrain, "virtual_obstacles")
  assert "edges" in cfg.scene.terrain.virtual_obstacles
  assert cfg.events["register_virtual_obstacles"].mode == "startup"


def test_task_registry_has_parkour_and_perceptive_tasks():
  import instinct_mjlab.tasks  # noqa: F401
  from instinct_mjlab.tasks.registry import list_tasks

  task_ids = set(list_tasks())
  assert "Instinct-Parkour-Target-Amp-G1-v0" in task_ids
  assert "Instinct-Parkour-Target-Amp-G1-Play-v0" in task_ids
  assert "Instinct-Perceptive-Shadowing-G1-v0" in task_ids
  assert "Instinct-Perceptive-Shadowing-G1-Play-v0" in task_ids
