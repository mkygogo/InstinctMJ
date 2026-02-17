from __future__ import annotations
from dataclasses import dataclass, field

from mjlab.sensor import RayCastSensorCfg


from .grouped_ray_caster import GroupedRayCaster


@dataclass(kw_only=True)
class GroupedRayCasterCfg(RayCastSensorCfg):
    """Configuration for the GroupedRayCaster sensor."""

    class_type: type = GroupedRayCaster

    min_distance: float = 0.0
    """The minimum distance from the sensor to ray cast to. aka ignore the hits closer than this distance."""

    aux_mesh_and_link_names: dict[str, str | None] = field(default_factory=dict)
    """The dictionary of merged mesh file (key) and link names (value). For the auxiliary mesh search when trying to dig
    out the mesh prim under a Xform prim. Please check all names where mesh file name is different from the link name.
    If the mesh file name is for the link name, set the `value` to None.
    For example, a torso link has a mesh file name `torso_link_rev_1_0`, but the link name is `torso_link`.
        It also has a `head_link` with a mesh file name `head_link` fixed in torso_link.
        Then, the `aux_mesh_and_link_names` should include:
        {
            "torso_link_rev_1_0": None,
            "head_link": "head_link",
        }
    """

    mesh_prim_paths: list[str] = field(default_factory=list)
    """Mesh path expressions kept for compatibility with the original GroupedRayCaster config layout."""

    attach_yaw_only: bool = False
    """Whether to apply only yaw when transforming ray starts (legacy IsaacLab behavior)."""

    def __post_init__(self):
        if self.attach_yaw_only:
            self.ray_alignment = "yaw"

    def build(self) -> GroupedRayCaster:
        return GroupedRayCaster(self)
