"""Instinct-RL configs for G1 perceptive shadowing tasks."""

from __future__ import annotations

from instinct_mjlab.rl import InstinctRlOnPolicyRunnerCfg

def g1_perceptive_shadowing_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
  return InstinctRlOnPolicyRunnerCfg(experiment_name="g1_perceptive_shadowing")


def g1_perceptive_vae_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
  return InstinctRlOnPolicyRunnerCfg(experiment_name="g1_perceptive_vae")
