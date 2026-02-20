from __future__ import annotations

import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING

from mjlab.terrains import SubTerrainBaseCfg, TerrainGenerator
from mjlab.terrains import TerrainImporter as TerrainImporterBase
from mjlab.utils.timer import Timer

if TYPE_CHECKING:
    from .terrain_importer_cfg import TerrainImporterCfg
    from .virtual_obstacle import VirtualObstacleBase


class TerrainImporter(TerrainImporterBase):
    def __init__(self, cfg: TerrainImporterCfg, device: str):
        self._virtual_obstacles = {}
        for name, virtual_obstacle_cfg in cfg.virtual_obstacles.items():
            if virtual_obstacle_cfg is None:
                continue
            virtual_obstacle = virtual_obstacle_cfg.class_type(virtual_obstacle_cfg)
            self._virtual_obstacles[name] = virtual_obstacle

        if cfg.terrain_type == "hacked_generator":
            self._hacked_terrain_type = "hacked_generator"
            cfg.terrain_type = "plane"
        super().__init__(cfg, device)

        # Keep flat-patch buffers aligned with generated terrain when hacked_generator is used.
        if getattr(self, "_hacked_terrain_type", None) == "hacked_generator" and self.terrain_generator is not None:
            self._flat_patches = {
                name: torch.from_numpy(arr).to(device=self.device, dtype=torch.float)
                for name, arr in self.terrain_generator.flat_patches.items()
            }
            self._flat_patch_radii = dict(self.terrain_generator.flat_patch_radii)

        terrain_mesh = self._get_terrain_mesh_for_virtual_obstacles()
        if terrain_mesh is not None:
            self._generate_virtual_obstacles(terrain_mesh)

    @property
    def virtual_obstacles(self) -> dict[str, VirtualObstacleBase]:
        """Get the virtual obstacles representing the edges.
        TODO: Make the returned value more general.
        """
        # still pointing the same VirtualObstacleBase objects but the dict is a copy.
        return self._virtual_obstacles.copy()

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg] | None:
        """Get the specific configurations for all subterrains."""
        # This is a placeholder. The actual implementation should return the specific configurations.
        return (
            self.terrain_generator.subterrain_specific_cfgs
            if hasattr(self, "terrain_generator") and hasattr(self.terrain_generator, "subterrain_specific_cfgs")
            else None
        )

    """
    Operations - Import.
    """

    def import_ground_plane(self, name: str):
        """
        NOTE:
        This is a hack to fit self-defined tasks and keep InstinctLab's control flow.
        For hacked_generator, terrain_type is routed through "plane" branch while still
        compiling custom generated terrain meshes.
        """
        if getattr(self, "_hacked_terrain_type", None) == "hacked_generator":
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'hacked_generator' but no value provided for 'terrain_generator'.")
            terrain_generator_cls = getattr(
                self.cfg.terrain_generator,
                "class_type",
                TerrainGenerator,
            )
            self.terrain_generator = terrain_generator_cls(
                self.cfg.terrain_generator,
                device=self.device,
            )
            self.terrain_generator.compile(self._spec)
            # Diagnostic: log terrain mesh geom count for debugging.
            terrain_body = self._spec.body("terrain")
            geom_count = sum(1 for _ in terrain_body.geoms)
            print(f"[TerrainImporter] hacked_generator compiled: {geom_count} geom(s) in 'terrain' body")
            if self.terrain_generator.terrain_origins is not None:
                print(f"[TerrainImporter] terrain_origins shape: {self.terrain_generator.terrain_origins.shape}")
            self.configure_env_origins(self.terrain_generator.terrain_origins)
        else:
            super().import_ground_plane(name)

    def _get_terrain_mesh_for_virtual_obstacles(self) -> trimesh.Trimesh | None:
        if self.terrain_generator is None:
            return None
        terrain_mesh = getattr(self.terrain_generator, "terrain_mesh", None)
        return terrain_mesh

    def _generate_virtual_obstacles(self, mesh: trimesh.Trimesh):
        """Generate virtual obstacles from a terrain mesh."""
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())  # remove duplicate faces
        mesh.remove_unreferenced_vertices()
        # Generate virtual obstacles based on the generated terrain mesh.
        # NOTE: generate virtual obstacle first because it might modify the mesh.
        for name, virtual_obstacle in self._virtual_obstacles.items():
            with Timer(f"Generate virtual obstacle {name}"):
                virtual_obstacle.generate(mesh, device=self.device)

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Set the debug visualization flag.

        Args:
            vis: True to enable debug visualization, False to disable.
        """
        results = True

        for name, virtual_obstacle in self._virtual_obstacles.items():
            if debug_vis:
                virtual_obstacle.visualize()
            else:
                virtual_obstacle.disable_visualizer()

        return results

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
        """Configure the environment origins.

        Args:
            origins: The origins of the environments. Shape is (num_envs, 3).
        """
        if origins is None and getattr(self, "_hacked_terrain_type", None) == "hacked_generator":
            # Keep terrain-based origins configured in hacked_generator path.
            return
        return super().configure_env_origins(origins)
