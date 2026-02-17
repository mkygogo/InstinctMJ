from dataclasses import dataclass
from mjlab.terrains import TerrainGeneratorCfg as TerrainGeneratorCfgBase

from .terrain_generator import FiledTerrainGenerator


@dataclass(kw_only=True)
class FiledTerrainGeneratorCfg(TerrainGeneratorCfgBase):
    class_type: type = FiledTerrainGenerator
