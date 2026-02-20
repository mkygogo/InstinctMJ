from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from mjlab.sensor import RayCastSensor

if TYPE_CHECKING:
    from .grouped_ray_caster_cfg import GroupedRayCasterCfg


class GroupedRayCaster(RayCastSensor):
    """Ray-caster with per-environment mutable ray offsets/directions."""

    cfg: GroupedRayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: GroupedRayCasterCfg):
        super().__init__(cfg)
        self._num_envs = 0
        self._ALL_INDICES = torch.empty(0, dtype=torch.long)
        self.drift: torch.Tensor | None = None
        self.ray_starts: torch.Tensor | None = None
        self.ray_directions: torch.Tensor | None = None

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)

        self._num_envs = data.nworld
        self._ALL_INDICES = torch.arange(self._num_envs, device=device, dtype=torch.long)
        self.drift = torch.zeros(self._num_envs, 3, device=device, dtype=torch.float32)

        assert self._local_offsets is not None and self._local_directions is not None
        self.ray_starts = self._local_offsets.unsqueeze(0).repeat(self._num_envs, 1, 1).clone()
        self.ray_directions = self._local_directions.unsqueeze(0).repeat(self._num_envs, 1, 1).clone()

    def prepare_rays(self) -> None:
        """PRE-GRAPH: Transform per-env local rays to world frame."""
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

        rot_mat = self._compute_alignment_rotation(frame_mat)
        world_offsets = torch.einsum("bij,bnj->bni", rot_mat, self.ray_starts)
        world_origins = frame_pos.unsqueeze(1) + world_offsets
        ray_directions_w = torch.einsum("bij,bnj->bni", rot_mat, self.ray_directions)

        if self.drift is not None:
            # apply drift
            world_origins = world_origins + self.drift.unsqueeze(1)
            frame_pos = frame_pos + self.drift

        pnt_torch = wp.to_torch(self._ray_pnt).view(self._num_envs, self._num_rays, 3)
        vec_torch = wp.to_torch(self._ray_vec).view(self._num_envs, self._num_rays, 3)
        pnt_torch.copy_(world_origins)
        vec_torch.copy_(ray_directions_w)

        self._cached_world_origins = world_origins
        self._cached_world_rays = ray_directions_w
        self._cached_frame_pos = frame_pos
        self._cached_frame_mat = frame_mat

    def postprocess_rays(self) -> None:
        super().postprocess_rays()
        if self.cfg.min_distance <= 0.0:
            return
        assert self._distances is not None
        assert self._normals_w is not None
        assert self._hit_pos_w is not None
        assert self._cached_world_origins is not None

        near_hit_mask = (self._distances >= 0.0) & (self._distances < self.cfg.min_distance)
        if not torch.any(near_hit_mask):
            return

        self._distances[near_hit_mask] = -1.0
        self._hit_pos_w[near_hit_mask] = self._cached_world_origins[near_hit_mask]
        self._normals_w[near_hit_mask] = 0.0
