"""Instinct-RL configs for G1 whole-body shadowing tasks."""

from __future__ import annotations

from instinct_mjlab.rl import InstinctRlOnPolicyRunnerCfg

from .agents.instinct_rl_ppo_cfg import g1_shadowing_ppo_runner_cfg


def g1_shadowing_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
  return g1_shadowing_ppo_runner_cfg()
