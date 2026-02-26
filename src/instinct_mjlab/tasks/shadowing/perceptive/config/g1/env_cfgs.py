"""G1 perceptive environment config adapters."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from .perceptive_shadowing_cfg import (
  G1PerceptiveShadowingEnvCfg,
  G1PerceptiveShadowingEnvCfg_PLAY,
)
from .perceptive_vae_cfg import (
  G1PerceptiveVaeEnvCfg,
  G1PerceptiveVaeEnvCfg_PLAY,
)


def instinct_g1_perceptive_shadowing_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return G1PerceptiveShadowingEnvCfg_PLAY() if play else G1PerceptiveShadowingEnvCfg()


def instinct_g1_perceptive_vae_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return G1PerceptiveVaeEnvCfg_PLAY() if play else G1PerceptiveVaeEnvCfg()


__all__ = [
  "G1PerceptiveShadowingEnvCfg",
  "G1PerceptiveShadowingEnvCfg_PLAY",
  "G1PerceptiveVaeEnvCfg",
  "G1PerceptiveVaeEnvCfg_PLAY",
  "instinct_g1_perceptive_shadowing_env_cfg",
  "instinct_g1_perceptive_vae_env_cfg",
]
