"""Register migrated InstinctLab perceptive G1 tasks."""

from instinct_mjlab.tasks.registry import register_instinct_task

from .env_cfgs import (
  instinct_g1_perceptive_shadowing_env_cfg,
  instinct_g1_perceptive_vae_env_cfg,
)
from .rl_cfgs import (
  g1_perceptive_shadowing_instinct_rl_cfg,
  g1_perceptive_vae_instinct_rl_cfg,
)


register_instinct_task(
  task_id="Instinct-Perceptive-Shadowing-G1-v0",
  env_cfg=instinct_g1_perceptive_shadowing_env_cfg(play=False),
  play_env_cfg=instinct_g1_perceptive_shadowing_env_cfg(play=True),
  instinct_rl_cfg=g1_perceptive_shadowing_instinct_rl_cfg(),
)

register_instinct_task(
  task_id="Instinct-Perceptive-Shadowing-G1-Play-v0",
  env_cfg=instinct_g1_perceptive_shadowing_env_cfg(play=True),
  play_env_cfg=instinct_g1_perceptive_shadowing_env_cfg(play=True),
  instinct_rl_cfg=g1_perceptive_shadowing_instinct_rl_cfg(),
)

register_instinct_task(
  task_id="Instinct-Perceptive-Vae-G1-v0",
  env_cfg=instinct_g1_perceptive_vae_env_cfg(play=False),
  play_env_cfg=instinct_g1_perceptive_vae_env_cfg(play=True),
  instinct_rl_cfg=g1_perceptive_vae_instinct_rl_cfg(),
)

register_instinct_task(
  task_id="Instinct-Perceptive-Vae-G1-Play-v0",
  env_cfg=instinct_g1_perceptive_vae_env_cfg(play=True),
  play_env_cfg=instinct_g1_perceptive_vae_env_cfg(play=True),
  instinct_rl_cfg=g1_perceptive_vae_instinct_rl_cfg(),
)
