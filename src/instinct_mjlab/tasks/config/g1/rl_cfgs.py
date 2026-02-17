"""Compatibility exports for legacy G1 Instinct-RL config imports."""

from instinct_mjlab.tasks.locomotion.config.g1.rl_cfgs import (
  g1_locomotion_instinct_rl_cfg,
)
from instinct_mjlab.tasks.parkour.config.g1.rl_cfgs import (
  g1_parkour_amp_instinct_rl_cfg,
)
from instinct_mjlab.tasks.shadowing.beyondmimic.config.g1.rl_cfgs import (
  g1_beyondmimic_instinct_rl_cfg,
)
from instinct_mjlab.tasks.shadowing.perceptive.config.g1.rl_cfgs import (
  g1_perceptive_shadowing_instinct_rl_cfg,
  g1_perceptive_vae_instinct_rl_cfg,
)
from instinct_mjlab.tasks.shadowing.whole_body.config.g1.rl_cfgs import (
  g1_shadowing_instinct_rl_cfg,
)

__all__ = [
  "g1_locomotion_instinct_rl_cfg",
  "g1_shadowing_instinct_rl_cfg",
  "g1_beyondmimic_instinct_rl_cfg",
  "g1_perceptive_shadowing_instinct_rl_cfg",
  "g1_perceptive_vae_instinct_rl_cfg",
  "g1_parkour_amp_instinct_rl_cfg",
]

