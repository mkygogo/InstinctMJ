from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv as ManagerBasedEnv


def register_virtual_obstacle_to_sensor(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_cfgs: list[SceneEntityCfg] | SceneEntityCfg,
    enable_debug_vis: bool = False,
):
    """Make each sensor accessible to the terrain virtual obstacle by providing `sensor.register_virtual_obstacles` with
    `terrain.virtual_obstacles` dict.

    """
    if isinstance(sensor_cfgs, SceneEntityCfg):
        sensor_cfgs = [sensor_cfgs]

    virtual_obstacles: dict = env.scene.terrain.virtual_obstacles

    for sensor_cfg in sensor_cfgs:
        sensor = env.scene[sensor_cfg.name]
        if not hasattr(sensor, "register_virtual_obstacles"):
            raise ValueError(f"Sensor {sensor_cfg.name} does not support virtual obstacles.")

        sensor.register_virtual_obstacles(virtual_obstacles)

    if enable_debug_vis:
        env.scene.terrain.set_debug_vis(True)
