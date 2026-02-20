from __future__ import annotations
from dataclasses import dataclass

from mjlab.sensor import RayCastSensorCfg


from .grouped_ray_caster import GroupedRayCaster


@dataclass(kw_only=True)
class GroupedRayCasterCfg(RayCastSensorCfg):
    """Configuration for the GroupedRayCaster sensor."""

    class_type: type = GroupedRayCaster

    min_distance: float = 0.0
    """The minimum distance from the sensor to ray cast to. aka ignore the hits closer than this distance."""

    def build(self) -> GroupedRayCaster:
        return GroupedRayCaster(self)
