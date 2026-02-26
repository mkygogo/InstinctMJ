"""Python package serving as the migrated InstinctLab module."""

from __future__ import annotations


def _patch_terrain_pipeline() -> None:
  """Route mjlab scene terrain creation through Instinct_mjlab terrain classes."""
  import mjlab.scene.scene as scene_module
  import mjlab.terrains as terrains_module
  import mjlab.terrains.terrain_entity as terrain_entity_module

  from instinct_mjlab.terrains.terrain_generator import FiledTerrainGenerator
  import instinct_mjlab.terrains.terrain_importer as instinct_terrain_importer_module
  from instinct_mjlab.terrains.terrain_importer import TerrainImporter as InstinctTerrainImporter

  scene_module.TerrainEntity = InstinctTerrainImporter
  terrain_entity_module.TerrainEntity = InstinctTerrainImporter
  terrain_entity_module.TerrainGenerator = FiledTerrainGenerator
  instinct_terrain_importer_module.TerrainGenerator = FiledTerrainGenerator
  terrains_module.TerrainEntity = InstinctTerrainImporter
  terrains_module.TerrainGenerator = FiledTerrainGenerator


def _patch_joint_action_order() -> None:
  """Make JOINT action preserve_order semantics match InstinctLab expectations."""
  import importlib

  entity_module = importlib.import_module("mjlab.entity.entity")
  actions_module = importlib.import_module("mjlab.envs.mdp.actions.actions")
  from mjlab.actuator.actuator import TransmissionType

  def _find_joints_by_actuator_names(
    self,
    actuator_name_keys,
    preserve_order: bool = False,
  ):
    actuated_joint_names_set = set()
    for act in self._actuators:
      actuated_joint_names_set.update(act.target_names)

    actuated_in_natural_order = [
      name for name in self.joint_names if name in actuated_joint_names_set
    ]
    _, matched_joint_names = self.find_joints(
      actuator_name_keys,
      joint_subset=actuated_in_natural_order,
      preserve_order=preserve_order,
    )
    name_to_entity_idx = {name: i for i, name in enumerate(self.joint_names)}
    joint_ids = [name_to_entity_idx[name] for name in matched_joint_names]
    return joint_ids, matched_joint_names

  if not getattr(entity_module.Entity, "_instinct_joint_order_patched", False):
    entity_module.Entity.find_joints_by_actuator_names = _find_joints_by_actuator_names
    entity_module.Entity._instinct_joint_order_patched = True

  if not getattr(actions_module.BaseAction, "_instinct_joint_order_patched", False):
    original_find_targets = actions_module.BaseAction._find_targets

    def _patched_find_targets(self, cfg):
      if cfg.transmission_type == TransmissionType.JOINT:
        return self._entity.find_joints_by_actuator_names(
          cfg.actuator_names,
          preserve_order=cfg.preserve_order,
        )
      return original_find_targets(self, cfg)

    actions_module.BaseAction._find_targets = _patched_find_targets
    actions_module.BaseAction._instinct_joint_order_patched = True


_patch_terrain_pipeline()
_patch_joint_action_order()
