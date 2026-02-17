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
):
    """Make each sensor accessible to the terrain virtual obstacle by providing `sensor.register_virtual_obstacles` with
    `terrain.virtual_obstacles` dict.

    """
    if isinstance(sensor_cfgs, SceneEntityCfg):
        sensor_cfgs = [sensor_cfgs]

    if env.scene.terrain is None:
        raise RuntimeError(
            "register_virtual_obstacle_to_sensor requires a terrain, but env.scene.terrain is None."
        )
    if not hasattr(env.scene.terrain, "virtual_obstacles"):
        raise TypeError(
            "register_virtual_obstacle_to_sensor requires terrain.virtual_obstacles support. "
            f"Got terrain type: {type(env.scene.terrain).__name__}."
        )
    virtual_obstacles: dict = env.scene.terrain.virtual_obstacles

    for sensor_cfg in sensor_cfgs:
        if sensor_cfg.name not in env.scene.sensors:
            raise KeyError(f"Sensor '{sensor_cfg.name}' is not registered in env.scene.sensors.")
        sensor = env.scene[sensor_cfg.name]
        if not hasattr(sensor, "register_virtual_obstacles"):
            raise TypeError(
                f"Sensor '{sensor_cfg.name}' does not implement register_virtual_obstacles()."
            )

        sensor.register_virtual_obstacles(virtual_obstacles)
