from dataclasses import dataclass, field
from typing import List

from mjlab.terrains.terrain_generator_cfg import SubTerrainBaseCfg

from ..height_field.hf_terrains_cfg import PerlinPlaneTerrainCfg
from . import mesh_terrains

@dataclass
class WallTerrainCfgMixin:
    wall_prob: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # Probability of generating walls on [left, right, front, back] sides
    wall_height: float = 5.0  # Height of the walls
    wall_thickness: float = 0.05  # Thickness of the walls

@dataclass(kw_only=True)
class MotionMatchedTerrainCfg(SubTerrainBaseCfg):
    """Configuration for motion-matched terrain generation.

    ## Terrain Mesh Requirements
    - All terrain meshes must have the a border at the bottom.
    - The terrain origin (0, 0, 0) is at the surface of the terrain center, which means that the point should
        be above the terrain at (0, 0, t) given any t > 0 and below the terrain at (0, 0, t) given any t < 0.
    - The USER should ensure that the non-flat part of the terrain is within the size of the terrain.
    """

    function: object = mesh_terrains.motion_matched_terrain


    path: str = None
    """Directory containing both terrains and the motions, so that these can be matched together.
    """

    metadata_yaml: str = None
    """YAML file containing the motion matching configuration.
    This file should specify the motion matching parameters, such as the motion files to be used,
    the matching criteria, and any other relevant settings.

    You may use the `scripts/motion_matched_metadata_generator.py` to generate the metadata.yaml file if you arrange your
    dataset in the structure as described in `scripts/motion_matched_metadata_generator.py`.

    ## Typical yaml file structure

    ```yaml
    terrains:
        - terrain_id: "jumpbox1" # can be any string.
          terrain_file: "path/to/terrain.stl" # path to the terrain mesh file, relative to the datasetdir.
        - terrain_id: "jumpbox2"
          terrain_file: "path/to/another_terrain.stl"
    motion_files:
        - terrain_id: "jumpbox1" # should match the terrain_id above.
          motion_file: "path/to/motion1_poses.npz" # path to the motion file, relative to the datasetdir.
          weight: (optional) 1.0
        - terrain_id: "jumpbox2"
          motion_file: "path/to/motion2_retargetted.npz"
          weight: (optional) 1.0
    ```

    """

@dataclass(kw_only=True)
class PerlinMeshFloatingBoxTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a floating box mesh terrain."""

    function: object = mesh_terrains.floating_box_terrain

    floating_height: tuple[float, float] | float = None
    """The height of the box above the ground. Could be a fixed value or a range (min, max)."""
    box_length: tuple[float, float] | float = None
    """The length of the box along the y-axis. Could be a fixed value or a range (min, max)."""
    box_width: float | None = None
    """The width of the box along the x-axis. If None, it will be equal to the width of the terrain."""
    box_height: tuple[float, float] | float = None
    """The height of the box along the z-axis."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    # values used for perlin noise generation
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    no_perlin_at_obstacle: bool = True
    """If True, no perlin noise will be generated exactly below the box."""

@dataclass(kw_only=True)
class PerlinMeshRandomMultiBoxTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a sub terrain with multiple random boxes with perlin noise."""

    function: object = mesh_terrains.random_multi_box_terrain

    box_height_mean: tuple[float, float] | float = None
    box_height_range: float = None
    box_length_mean: tuple[float, float] | float = None
    box_length_range: float = None
    box_width_mean: tuple[float, float] | float = None
    box_width_range: float = None
    platform_width: float = None

    generation_ratio: float = None

    perlin_cfg: PerlinPlaneTerrainCfg | None = None
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    no_perlin_at_obstacle: bool = False
    box_perlin_cfg: PerlinPlaneTerrainCfg | None = None
    """Used only when perlin_cfg is not None"""
