"""G1 BeyondMimic environment config adapters."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from .beyondmimic_plane_cfg import (
  G1BeyondMimicPlaneEnvCfg,
  G1BeyondMimicPlaneEnvCfg_PLAY,
)


def instinct_g1_beyondmimic_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return G1BeyondMimicPlaneEnvCfg_PLAY() if play else G1BeyondMimicPlaneEnvCfg()


__all__ = [
  "G1BeyondMimicPlaneEnvCfg",
  "G1BeyondMimicPlaneEnvCfg_PLAY",
  "instinct_g1_beyondmimic_env_cfg",
]
