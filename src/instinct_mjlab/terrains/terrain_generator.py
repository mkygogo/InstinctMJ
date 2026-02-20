from __future__ import annotations

import copy
import inspect
import mujoco
import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING

from mjlab.terrains import SubTerrainBaseCfg, TerrainGenerator

if TYPE_CHECKING:
    from .terrain_generator_cfg import FiledTerrainGeneratorCfg


class FiledTerrainGenerator(TerrainGenerator):
    """A terrain generator that uses the filed generator."""

    def __init__(self, cfg: FiledTerrainGeneratorCfg, device: str = "cpu"):

        # Access the i-th row, j-th column subterrain config by
        # self._subterrain_specific_cfgs[i*num_cols + j]
        self._subterrain_specific_cfgs: list[SubTerrainBaseCfg] = []
        self._terrain_meshes: list[trimesh.Trimesh] = []
        self.terrain_mesh: trimesh.Trimesh | None = None
        super().__init__(cfg, device)

    def compile(self, spec: mujoco.MjSpec) -> None:
        self._terrain_meshes = []
        self.terrain_mesh = None
        super().compile(spec)
        if len(self._terrain_meshes) == 1:
            self.terrain_mesh = self._terrain_meshes[0]
        elif len(self._terrain_meshes) > 1:
            self.terrain_mesh = trimesh.util.concatenate(self._terrain_meshes)

    def _get_subterrain_function(self, cfg: SubTerrainBaseCfg):
        terrain_function = inspect.getattr_static(type(cfg), "function")
        if isinstance(terrain_function, (staticmethod, classmethod)):
            terrain_function = terrain_function.__func__
        return terrain_function

    def _create_legacy_terrain_geom(
        self,
        spec: mujoco.MjSpec,
        world_position: np.ndarray,
        meshes: trimesh.Trimesh | list[trimesh.Trimesh] | tuple[trimesh.Trimesh, ...],
        origin: np.ndarray,
        sub_row: int,
        sub_col: int,
    ) -> np.ndarray:
        if isinstance(meshes, trimesh.Trimesh):
            meshes_list = [meshes]
        elif isinstance(meshes, (list, tuple)):
            meshes_list = list(meshes)
        else:
            raise TypeError(
                "Legacy terrain function must return a trimesh.Trimesh or a list/tuple of trimesh.Trimesh."
            )

        body = spec.body("terrain")
        for mesh_idx, mesh in enumerate(meshes_list):
            if not isinstance(mesh, trimesh.Trimesh):
                raise TypeError("Legacy terrain function returned a non-trimesh mesh entry.")
            mesh_name = f"terrain_mesh_{sub_row}_{sub_col}_{mesh_idx}"
            spec.add_mesh(
                name=mesh_name,
                uservert=np.asarray(mesh.vertices, dtype=np.float32).reshape(-1).tolist(),
                userface=np.asarray(mesh.faces, dtype=np.int32).reshape(-1).tolist(),
            )
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                pos=world_position,
            )
            if self.cfg.color_scheme == "random":
                geom.rgba[:3] = self.np_rng.uniform(0.3, 0.8, 3)
                geom.rgba[3] = 1.0
            elif self.cfg.color_scheme == "none":
                geom.rgba[:] = (0.5, 0.5, 0.5, 1.0)

            # Keep a world-frame terrain mesh for virtual obstacle generation.
            world_mesh = mesh.copy()
            world_mesh.apply_translation(world_position)
            self._terrain_meshes.append(world_mesh)

        spawn_origin = np.asarray(origin, dtype=np.float64) + world_position
        for _, arr in self.flat_patches.items():
            # Legacy terrains do not provide flat patches; keep the super implementation's fallback behavior.
            arr[sub_row, sub_col] = spawn_origin
        return spawn_origin

    def _create_terrain_geom(
        self,
        spec: mujoco.MjSpec,
        world_position: np.ndarray,
        difficulty: float,
        cfg: SubTerrainBaseCfg,
        sub_row: int,
        sub_col: int,
    ):
        """This function intercept the terrain mesh generation process and records the specific config
        for each subterrain.
        """
        terrain_function = self._get_subterrain_function(cfg)
        num_args = len(inspect.signature(terrain_function).parameters)
        if num_args == 2:
            meshes, origin = terrain_function(difficulty, cfg)
            spawn_origin = self._create_legacy_terrain_geom(
                spec, world_position, meshes, origin, sub_row, sub_col
            )
        elif num_args == 4:
            # Record mesh names before calling super so we can identify newly-added mesh geoms.
            mesh_names_before = {m.name for m in spec.meshes}
            spawn_origin = super()._create_terrain_geom(
                spec,
                world_position,
                difficulty,
                cfg,
                sub_row,
                sub_col,
            )
            # Collect world-frame mesh for virtual obstacle generation (mirrors legacy path).
            new_mesh_names = {m.name for m in spec.meshes} - mesh_names_before
            for geom in spec.body("terrain").geoms:
                mesh_name = getattr(geom, "meshname", "")
                if not isinstance(mesh_name, str) or mesh_name not in new_mesh_names:
                    continue
                mjs_mesh = spec.mesh(mesh_name)
                if mjs_mesh is None:
                    continue
                verts = np.array(mjs_mesh.uservert, dtype=np.float32).reshape(-1, 3)
                faces = np.array(mjs_mesh.userface, dtype=np.int32).reshape(-1, 3)
                geom_pos = np.array(geom.pos, dtype=np.float64)
                world_mesh = trimesh.Trimesh(vertices=verts + geom_pos, faces=faces, process=False)
                self._terrain_meshes.append(world_mesh)
        else:
            raise TypeError(
                f"Unsupported terrain function signature for {type(cfg).__name__}: "
                f"expected 2 (legacy) or 4 (mjlab) arguments, got {num_args}."
            )
        # >>> NOTE: This code snippet is copied from the super implementation because they copied the cfg
        # but we need to store the modified cfg for each subterrain.
        cfg = copy.deepcopy(cfg)
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # <<< NOTE
        self._subterrain_specific_cfgs.append(cfg)  # since in super function, cfg is a copy of the original config.

        return spawn_origin

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg]:
        """Get the specific configurations for all subterrains."""
        return self._subterrain_specific_cfgs.copy()  # Return a copy to avoid external modification.

    def get_subterrain_cfg(
        self, row_ids: int | torch.Tensor, col_ids: int | torch.Tensor
    ) -> list[SubTerrainBaseCfg] | SubTerrainBaseCfg | None:
        """Get the specific configuration for a subterrain by its row and column index."""
        num_cols = self.cfg.num_cols
        idx = row_ids * num_cols + col_ids
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy().tolist()  # Convert to list if it's a tensor.
            return [
                self._subterrain_specific_cfgs[i] if 0 <= i < len(self._subterrain_specific_cfgs) else None for i in idx
            ]
        if isinstance(idx, int):
            return self._subterrain_specific_cfgs[idx] if 0 <= idx < len(self._subterrain_specific_cfgs) else None
