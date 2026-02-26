from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from mjlab.viewer.viewer_config import ViewerConfig


@dataclass
class InstinctLabViewerConfig(ViewerConfig):
  """Instinct-specific viewer config keeping original InstinctLab-style params.

  Built on top of mjlab ``ViewerConfig`` while preserving legacy fields:
  - ``origin_type`` string-style enum values
  - ``asset_name`` alias for ``entity_name``
  - ``eye`` (camera position) mapped to spherical camera params
  - ``resolution`` mapped to ``width`` / ``height``
  """

  class OriginType(str, enum.Enum):
    WORLD = "world"
    ASSET_ROOT = "asset_root"
    ASSET_BODY = "asset_body"

  origin_type: OriginType | str = OriginType.WORLD
  asset_name: str | None = None
  eye: tuple[float, float, float] | None = None
  resolution: tuple[int, int] | None = None
  debug_vis_show_all_envs: bool = False

  def __post_init__(self) -> None:
    self.origin_type = self._coerce_origin_type(self.origin_type)

    if self.asset_name is not None:
      self.entity_name = self.asset_name
    elif self.entity_name is not None:
      self.asset_name = self.entity_name

    if self.resolution is None:
      self.resolution = (int(self.width), int(self.height))
    else:
      self.width = int(self.resolution[0])
      self.height = int(self.resolution[1])
      self.resolution = (int(self.width), int(self.height))

    if self.eye is None:
      self.eye = self._compute_eye_from_spherical()
    else:
      self._sync_spherical_from_eye(self.eye)
      self.eye = tuple(float(v) for v in self.eye)

  def __setattr__(self, name: str, value) -> None:
    object.__setattr__(self, name, value)

    if name == "origin_type":
      object.__setattr__(self, "origin_type", self._coerce_origin_type(value))
      return

    if name == "asset_name":
      object.__setattr__(self, "entity_name", value)
      return

    if name == "entity_name":
      object.__setattr__(self, "asset_name", value)
      return

    if name == "resolution" and value is not None:
      width, height = int(value[0]), int(value[1])
      object.__setattr__(self, "width", width)
      object.__setattr__(self, "height", height)
      object.__setattr__(self, "resolution", (width, height))
      return

    if name in {"width", "height"} and self._has_attrs("width", "height"):
      object.__setattr__(self, "resolution", (int(self.width), int(self.height)))
      return

    if name == "eye" and value is not None:
      eye_tuple = tuple(float(v) for v in value)
      object.__setattr__(self, "eye", eye_tuple)
      if self._has_attrs("lookat"):
        self._sync_spherical_from_eye(eye_tuple)
      return

    if name in {"lookat", "distance", "elevation", "azimuth"}:
      if self._has_attrs("lookat", "distance", "elevation", "azimuth"):
        object.__setattr__(self, "eye", self._compute_eye_from_spherical())

  @staticmethod
  def _coerce_origin_type(value) -> OriginType | str:
    if isinstance(value, InstinctLabViewerConfig.OriginType):
      return value
    if isinstance(value, str):
      key = value.strip().lower()
      mapping = {
        "world": InstinctLabViewerConfig.OriginType.WORLD,
        "asset_root": InstinctLabViewerConfig.OriginType.ASSET_ROOT,
        "asset_body": InstinctLabViewerConfig.OriginType.ASSET_BODY,
      }
      return mapping.get(key, value)
    if hasattr(value, "name"):
      key = str(value.name).strip().lower()
      mapping = {
        "world": InstinctLabViewerConfig.OriginType.WORLD,
        "asset_root": InstinctLabViewerConfig.OriginType.ASSET_ROOT,
        "asset_body": InstinctLabViewerConfig.OriginType.ASSET_BODY,
      }
      return mapping.get(key, value)
    return value

  def _has_attrs(self, *names: str) -> bool:
    return all(name in self.__dict__ for name in names)

  def _compute_eye_from_spherical(self) -> tuple[float, float, float]:
    lookat = self.lookat
    distance = float(self.distance)
    azimuth_rad = math.radians(float(self.azimuth))
    elevation_rad = math.radians(float(self.elevation))
    x = lookat[0] + distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = lookat[1] + distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = lookat[2] + distance * math.sin(elevation_rad)
    return (x, y, z)

  def _sync_spherical_from_eye(self, eye: tuple[float, float, float]) -> None:
    dx = float(eye[0]) - float(self.lookat[0])
    dy = float(eye[1]) - float(self.lookat[1])
    dz = float(eye[2]) - float(self.lookat[2])
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance <= 1e-9:
      return
    elevation = math.degrees(math.asin(dz / distance))
    azimuth = math.degrees(math.atan2(dy, dx))
    object.__setattr__(self, "distance", float(distance))
    object.__setattr__(self, "elevation", float(elevation))
    object.__setattr__(self, "azimuth", float(azimuth))

