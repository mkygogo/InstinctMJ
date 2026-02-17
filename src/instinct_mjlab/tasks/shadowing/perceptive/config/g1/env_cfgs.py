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
  return (
    G1PerceptiveShadowingEnvCfg_PLAY(decimation=4)
    if play
    else G1PerceptiveShadowingEnvCfg(decimation=4)
  )


def instinct_g1_perceptive_vae_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return G1PerceptiveVaeEnvCfg_PLAY(decimation=4) if play else G1PerceptiveVaeEnvCfg(decimation=4)


__all__ = [
  "G1PerceptiveShadowingEnvCfg",
  "G1PerceptiveShadowingEnvCfg_PLAY",
  "G1PerceptiveVaeEnvCfg",
  "G1PerceptiveVaeEnvCfg_PLAY",
  "instinct_g1_perceptive_shadowing_env_cfg",
  "instinct_g1_perceptive_vae_env_cfg",
]
