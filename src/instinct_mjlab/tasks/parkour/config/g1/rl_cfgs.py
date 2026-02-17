"""Instinct-RL configs for G1 parkour AMP tasks."""

from __future__ import annotations

from .agents.instinct_rl_amp_cfg import G1ParkourPPORunnerCfg


def g1_parkour_amp_instinct_rl_cfg() -> G1ParkourPPORunnerCfg:
  return G1ParkourPPORunnerCfg()
