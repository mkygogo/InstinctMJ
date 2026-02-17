"""Instinct-RL configs for G1 BeyondMimic tasks."""

from __future__ import annotations

from instinct_mjlab.rl import InstinctRlOnPolicyRunnerCfg

from .agents.beyondmimic_ppo_cfg import g1_beyondmimic_ppo_runner_cfg


def g1_beyondmimic_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
  return g1_beyondmimic_ppo_runner_cfg()
