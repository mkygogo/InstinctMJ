from __future__ import annotations
from dataclasses import dataclass


from .aistpp_motion import AistppMotion
from .amass_motion_cfg import AmassMotionCfg


@dataclass(kw_only=True)
class AistppMotionCfg(AmassMotionCfg):
    """Configuration for AIST++ motion files."""

    class_type: type = AistppMotion
