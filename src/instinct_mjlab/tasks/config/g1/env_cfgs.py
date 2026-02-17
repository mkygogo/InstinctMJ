"""Compatibility exports for legacy G1 environment config imports."""

from instinct_mjlab.tasks.locomotion.config.g1.flat_env_cfg import (
  instinct_g1_locomotion_flat_env_cfg,
)
from instinct_mjlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import (
  instinct_g1_parkour_amp_env_cfg,
)
from instinct_mjlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import (
  G1ParkourEnvCfg,
  G1ParkourEnvCfg_PLAY,
)
from instinct_mjlab.tasks.shadowing.beyondmimic.config.g1.beyondmimic_plane_cfg import (
  G1BeyondMimicPlaneEnvCfg,
  G1BeyondMimicPlaneEnvCfg_PLAY,
)
from instinct_mjlab.tasks.shadowing.beyondmimic.config.g1.env_cfgs import (
  instinct_g1_beyondmimic_env_cfg,
)
from instinct_mjlab.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_cfg import (
  G1PerceptiveShadowingEnvCfg,
  G1PerceptiveShadowingEnvCfg_PLAY,
)
from instinct_mjlab.tasks.shadowing.perceptive.config.g1.perceptive_vae_cfg import (
  G1PerceptiveVaeEnvCfg,
  G1PerceptiveVaeEnvCfg_PLAY,
)
from instinct_mjlab.tasks.shadowing.perceptive.config.g1.env_cfgs import (
  instinct_g1_perceptive_shadowing_env_cfg,
  instinct_g1_perceptive_vae_env_cfg,
)
from instinct_mjlab.tasks.shadowing.whole_body.config.g1.plane_shadowing_cfg import (
  G1PlaneShadowingEnvCfg,
  G1PlaneShadowingEnvCfg_PLAY,
)

__all__ = [
  "instinct_g1_locomotion_flat_env_cfg",
  "G1ParkourEnvCfg",
  "G1ParkourEnvCfg_PLAY",
  "instinct_g1_parkour_amp_env_cfg",
  "G1BeyondMimicPlaneEnvCfg",
  "G1BeyondMimicPlaneEnvCfg_PLAY",
  "instinct_g1_beyondmimic_env_cfg",
  "G1PerceptiveShadowingEnvCfg",
  "G1PerceptiveShadowingEnvCfg_PLAY",
  "G1PerceptiveVaeEnvCfg",
  "G1PerceptiveVaeEnvCfg_PLAY",
  "instinct_g1_perceptive_shadowing_env_cfg",
  "instinct_g1_perceptive_vae_env_cfg",
  "G1PlaneShadowingEnvCfg",
  "G1PlaneShadowingEnvCfg_PLAY",
]
