"""Shared helpers for Instinct-RL task configs."""

from __future__ import annotations

from instinct_mjlab.rl import InstinctRlNormalizerCfg


def default_policy_critic_normalizers() -> dict[str, InstinctRlNormalizerCfg]:
  return {
    "policy": InstinctRlNormalizerCfg(),
    "critic": InstinctRlNormalizerCfg(),
  }
