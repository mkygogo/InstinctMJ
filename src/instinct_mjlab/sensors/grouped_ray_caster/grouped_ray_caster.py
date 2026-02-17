from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from mjlab.sensor import RayCastData, RayCastSensor

if TYPE_CHECKING:
    from .grouped_ray_caster_cfg import GroupedRayCasterCfg


@dataclass
class _SensorView:
    """A minimal compatibility view object with only ``count`` field."""

    count: int = 0


class GroupedRayCaster(RayCastSensor):
    """Grouped Ray Caster sensor reads multiple meshes and keeps ray groups per environment."""

    cfg: GroupedRayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: GroupedRayCasterCfg):
        super().__init__(cfg)
        self.meshes: dict[str, list[Any]] = dict()  # {prim_path: [warp_meshes]}
        self.mesh_transforms: torch.Tensor | None = None  # shape (N, 4, 4)
        self.mesh_inv_transforms: torch.Tensor | None = None  # shape (N, 4, 4)
        self.mesh_prototype_ids: torch.Tensor | list[int] = []  # int64 shape (N,)
        self.mesh_collision_groups: torch.Tensor | list[int] = []  # int32 shape (N,)
        self.rigid_body_mesh_transform_segments: dict[str, slice] = dict()
        self.rigid_body_views: dict[str, Any] = dict()  # {prim_path: rigid_body_view}

        self._view = _SensorView()
        self._ALL_INDICES = torch.empty(0, dtype=torch.long)
        self.drift: torch.Tensor | None = None
        self.ray_starts: torch.Tensor | None = None
        self.ray_directions: torch.Tensor | None = None

        self._ray_collision_groups: torch.Tensor | None = None
        self._mesh_ids_for_group: torch.Tensor | None = None
        self._mesh_ids_slice_for_group: torch.Tensor | None = None

        self.mesh_transforms_pyt: torch.Tensor | None = None
        self.mesh_inv_transforms_pyt: torch.Tensor | None = None
        self.mesh_collision_groups_pyt: torch.Tensor | None = None
        self.mesh_prototype_ids_pyt: torch.Tensor | None = None
        self.all_mesh_indices: torch.Tensor | None = None

    @property
    def device(self) -> str:
        """Compatibility alias to match the original IsaacLab sensor fields."""
        assert self._device is not None
        return self._device

    @property
    def num_instances(self) -> int:
        """Compatibility alias to match IsaacLab sensor API."""
        return self._view.count

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)

        self._view = _SensorView(count=data.nworld)
        self._ALL_INDICES = torch.arange(self._view.count, device=device, dtype=torch.long)
        self.drift = torch.zeros(self._view.count, 3, device=device, dtype=torch.float32)

        assert self._local_offsets is not None and self._local_directions is not None
        self.ray_starts = self._local_offsets.unsqueeze(0).repeat(self._view.count, 1, 1).clone()
        self.ray_directions = self._local_directions.unsqueeze(0).repeat(self._view.count, 1, 1).clone()

        self._initialize_warp_meshes()
        self._create_ray_collision_groups()

    def _initialize_warp_meshes(self):
        """Initialize mesh-related buffers for grouped ray casting bookkeeping.

        ### NOTE (original IsaacLab)
            This is a re-implementation of the `RayCaster._initialize_warp_meshes` method.
            This should support multiple meshes and update their positions.
            Use env_ids to specify the collision group ids for the ray caster.

        Basic insights of getting rigid bodies mesh (original):
            prim_utils.get_prim_at_path(
                '/World/envs/env_0/Robot/torso_link/visuals/{torso_link_file_name}/mesh'
            ).IsA(UsdGeom.Mesh)

            However, {torso_link_file_name} could be different from the link name.
            This needs self.cfg.aux_mesh_and_link_names to be set.

        ### NOTE (mjlab migration)
            In mjlab, the BVH and geom transforms are managed by mujoco_warp internally.
            We keep these buffers for API compatibility with the original GroupedRayCaster.
        """
        self.meshes = {}
        self.mesh_prototype_ids = [0]
        self.mesh_collision_groups = [-1]

        assert self._device is not None
        self.mesh_transforms_pyt = torch.zeros(1, 7, dtype=torch.float32, device=self._device)
        self.mesh_transforms_pyt[:, -1] = 1.0
        self.mesh_inv_transforms_pyt = torch.zeros_like(self.mesh_transforms_pyt)
        self.mesh_inv_transforms_pyt[:, -1] = 1.0
        self.mesh_collision_groups_pyt = torch.tensor(self.mesh_collision_groups, dtype=torch.int32, device=self._device)
        self.mesh_prototype_ids_pyt = torch.tensor(self.mesh_prototype_ids, dtype=torch.int64, device=self._device)
        self.all_mesh_indices = torch.arange(len(self.mesh_prototype_ids), dtype=torch.int32, device=self._device)

    def _create_ray_collision_groups(self):
        """Create buffer to store ray collision groups and mesh ids for group ids."""
        if self.ray_starts is None:
            return

        assert self._device is not None
        self._ray_collision_groups = (
            torch.arange(self._view.count, dtype=torch.int32, device=self._device)
            .unsqueeze(1)
            .repeat(1, self.num_rays)
        )
        unique_groups = torch.unique(self._ray_collision_groups)
        # NOTE: For code consistency, we do not put for following code in a separate function, since unique_groups is
        # only acquired from self._ray_collision_groups currently.
        mesh_ids_for_group = []
        mesh_ids_slice_for_group = []

        assert self.mesh_collision_groups_pyt is not None
        for group_id in unique_groups:
            negative_one_indices = torch.where(self.mesh_collision_groups_pyt == -1)[0]
            group_indices = torch.where(self.mesh_collision_groups_pyt == group_id)[0]
            ray_group = torch.cat([negative_one_indices, group_indices]).tolist()
            mesh_ids_for_group.append(ray_group)
            mesh_ids_slice_for_group.append(len(ray_group))

        self._mesh_ids_for_group = torch.tensor(mesh_ids_for_group, dtype=torch.int32, device=self._device).view(-1)
        mesh_ids_slice_for_group = [0] + [
            sum(mesh_ids_slice_for_group[: i + 1]) for i in range(len(mesh_ids_slice_for_group))
        ]
        self._mesh_ids_slice_for_group = torch.tensor(mesh_ids_slice_for_group, dtype=torch.int32, device=self._device)

    def _update_mesh_transforms(self, env_ids: torch.Tensor | None = None):
        """Update the mesh transforms for the given environment IDs.
        This will update the mesh transforms based on the rigid body views.
        """
        # mujoco_warp updates BVH transforms internally; no extra mesh transform updates are required.
        del env_ids

    def prepare_rays(self) -> None:
        """PRE-GRAPH: Transform local rays to world frame with grouped-ray compatibility."""
        assert self._data is not None
        assert self.ray_starts is not None and self.ray_directions is not None
        assert self._ray_pnt is not None and self._ray_vec is not None

        if self._frame_type == "body":
            assert self._frame_body_id is not None
            frame_pos = self._data.xpos[:, self._frame_body_id]
            frame_mat = self._data.xmat[:, self._frame_body_id].view(-1, 3, 3)
        elif self._frame_type == "site":
            assert self._frame_site_id is not None
            frame_pos = self._data.site_xpos[:, self._frame_site_id]
            frame_mat = self._data.site_xmat[:, self._frame_site_id].view(-1, 3, 3)
        else:  # geom
            assert self._frame_geom_id is not None
            frame_pos = self._data.geom_xpos[:, self._frame_geom_id]
            frame_mat = self._data.geom_xmat[:, self._frame_geom_id].view(-1, 3, 3)

        # note: we clone here because we are read-only operations
        frame_pos = frame_pos.clone()
        frame_mat = frame_mat.clone()

        # ray cast based on the sensor poses
        if self.cfg.attach_yaw_only:
            # only yaw orientation is considered and directions are not rotated
            rot_mat = self._extract_yaw_rotation(frame_mat)
            world_offsets = torch.einsum("bij,bnj->bni", rot_mat, self.ray_starts)
            world_origins = frame_pos.unsqueeze(1) + world_offsets
            ray_directions_w = self.ray_directions
        else:
            # full orientation is considered
            rot_mat = self._compute_alignment_rotation(frame_mat)
            world_offsets = torch.einsum("bij,bnj->bni", rot_mat, self.ray_starts)
            world_origins = frame_pos.unsqueeze(1) + world_offsets
            ray_directions_w = torch.einsum("bij,bnj->bni", rot_mat, self.ray_directions)

        if self.drift is not None:
            # apply drift
            world_origins = world_origins + self.drift.unsqueeze(1)
            frame_pos = frame_pos + self.drift

        num_envs = frame_pos.shape[0]
        pnt_torch = wp.to_torch(self._ray_pnt).view(num_envs, self._num_rays, 3)
        vec_torch = wp.to_torch(self._ray_vec).view(num_envs, self._num_rays, 3)
        pnt_torch.copy_(world_origins)
        vec_torch.copy_(ray_directions_w)

        self._cached_world_origins = world_origins
        self._cached_world_rays = ray_directions_w
        self._cached_frame_pos = frame_pos
        self._cached_frame_mat = frame_mat

    def _compute_data(self) -> RayCastData:
        data = super()._compute_data()
        # Compatibility aliases expected by migrated IsaacLab-style code.
        setattr(data, "ray_hits_w", data.hit_pos_w)
        return data
