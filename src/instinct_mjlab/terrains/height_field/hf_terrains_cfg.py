from dataclasses import dataclass, field
from typing import List

from mjlab.terrains.height_field import (
    HfDiscreteObstaclesTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfSteppingStonesTerrainCfg,
    HfTerrainBaseCfg,
    HfWaveTerrainCfg,
)

from . import hf_terrains

@dataclass
class WallTerrainCfgMixin:
    border_width: float = 0.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    wall_prob: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # Probability of generating walls on [left, right, front, back] sides
    wall_height: float = 5.0  # Height of the walls
    wall_thickness: float = 0.05  # Thickness of the walls

@dataclass(kw_only=True)
class PerlinPlaneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_plane_terrain

    noise_scale: float | List[float] = 0.05
    noise_frequency: int = 20

    fractal_octaves: int = 2

    fractal_lacunarity: float = 2.0

    fractal_gain: float = 0.25

    centering: bool = False  # If True, the noise will be centered around 0

@dataclass(kw_only=True)
class PerlinPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_pyramid_sloped_terrain

    slope_range: tuple[float, float] = None
    platform_width: float = 1.0
    inverted: bool = False
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinInvertedPyramidSlopedTerrainCfg(HfInvertedPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_pyramid_sloped_terrain

    slope_range: tuple[float, float] = None
    platform_width: float = 1.0
    inverted: bool = True
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_pyramid_stairs_terrain

    step_height_range: tuple[float, float] = None
    step_width: float = None
    platform_width: float = 1.0
    inverted: bool = False
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinInvertedPyramidStairsTerrainCfg(HfInvertedPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_pyramid_stairs_terrain

    step_height_range: tuple[float, float] = None
    step_width: float = None
    platform_width: float = 1.0
    inverted: bool = True
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinDiscreteObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_discrete_obstacles_terrain

    obstacle_height_mode: str = "choice"
    obstacle_width_range: tuple[float, float] = None
    obstacle_height_range: tuple[float, float] = None
    num_obstacles: int = None
    platform_width: float = 1.0
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinWaveTerrainCfg(HfWaveTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_wave_terrain

    amplitude_range: tuple[float, float] = None
    num_waves: int = 1
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinSteppingStonesTerrainCfg(HfSteppingStonesTerrainCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_stepping_stones_terrain

    stone_height_max: float = None
    stone_width_range: tuple[float, float] = None
    stone_distance_range: tuple[float, float] = None
    holes_depth: float = -10.0
    platform_width: float = 1.0
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

# -- Newly added terrain configurations for parkour terrains-- #
@dataclass(kw_only=True)
class PerlinParapetTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a parapet terrain, can be used for jump and hurdle tasks."""

    function: object = hf_terrains.perlin_parapet_terrain

    parapet_height: tuple[float, float] | float = (0.1, 0.3)
    parapet_length: tuple[float, float] | float = (0.1, 0.3)
    parapet_width: float | None = None
    curved_top_rate: float | None = None
    """The rate to generate curved top. If None, the top will be flat."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinGutterTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a gutter parkour terrain."""

    function: object = hf_terrains.perlin_gutter_terrain

    gutter_length: tuple[float, float] | float = (0.5, 1.5)  # the distance between gutters
    gutter_depth: tuple[float, float] | float = (0.1, 0.3)  # the depth of the gutter
    gutter_width: float | None = None  # the length of the gutter
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinStairsUpDownTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs up and down parkour terrain."""

    function: object = hf_terrains.perlin_stairs_up_down_terrain

    per_step_height: tuple[float, float] | float = None
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = None
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = None
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinStairsDownUpTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs down and up parkour terrain."""

    function: object = hf_terrains.perlin_stairs_down_up_terrain

    per_step_height: tuple[float, float] | float = None
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = None
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = None
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinTiltTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilt terrain."""

    function: object = hf_terrains.perlin_tilt_terrain

    wall_height: tuple[float, float] | float = None
    wall_width: float | None = None
    wall_length: tuple[float, float] | float = None
    wall_opening_angle: tuple[float, float] | float = None  # in degrees
    wall_opening_width: tuple[float, float] | float = None
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinTiltedRampTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilted ramp terrain."""

    function: object = hf_terrains.perlin_tilted_ramp_terrain

    tilt_angle: tuple[float, float] | float = None  # in degrees
    tilt_height: tuple[float, float] | float = None
    tilt_width: tuple[float, float] | float = None
    tilt_length: tuple[float, float] | float = None
    switch_spacing: tuple[float, float] | float = None
    spacing_curriculum: bool | None = None
    overlap_size: float | None = None
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinSlopeTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a slope up and down terrain with a flat ground in the middle."""

    function: object = hf_terrains.perlin_slope_terrain

    slope_angle: tuple[float, float] | float = None  # in degrees
    per_slope_length: tuple[float, float] | float = None
    platform_length: float = 1.0
    slope_width: float | None = None
    up_down: bool | None = None  # If True or None, the slope will be up and down, otherwise it will be down and up.
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinCrossStoneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a cross stone terrain."""

    function: object = hf_terrains.perlin_cross_stone_terrain

    stone_size: tuple[float, float] = None
    stone_height: tuple[float, float] | float = None
    stone_spacing: tuple[float, float] | float = None
    ground_depth: float = -0.5
    platform_width: float = 1.5
    xy_random_ratio: float = 0.2
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinSquareGapTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    function: object = hf_terrains.perlin_square_gap_terrain

    gap_distance_range: tuple[float, float] = (0.1, 0.5)
    gap_depth: tuple[float, float] = (0.2, 0.5)
    platform_width: float = 1.5
    border_width: float = 0.0

    perlin_cfg: PerlinPlaneTerrainCfg | None = None
