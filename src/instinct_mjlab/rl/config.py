"""Instinct-RL configuration dataclasses for mjlab integration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


def _to_plain_dict(obj: Any) -> Any:
  """Recursively convert dataclass objects into plain dictionaries."""
  if is_dataclass(obj):
    result = {}
    for item in fields(obj):
      value = getattr(obj, item.name)
      if value is None:
        continue
      result[item.name] = _to_plain_dict(value)
    return result
  if isinstance(obj, dict):
    return {key: _to_plain_dict(value) for key, value in obj.items()}
  if isinstance(obj, tuple):
    return [_to_plain_dict(value) for value in obj]
  if isinstance(obj, list):
    return [_to_plain_dict(value) for value in obj]
  return obj


@dataclass
class InstinctRlActorCriticCfg:
  class_name: str = "ActorCritic"
  init_noise_std: float = 1.0
  actor_hidden_dims: tuple[int, ...] = (256, 128, 128)
  critic_hidden_dims: tuple[int, ...] = (256, 128, 128)
  activation: str = "elu"
  num_moe_experts: int | None = None
  moe_gate_hidden_dims: tuple[int, ...] | None = None
  encoder_configs: dict[str, Any] | None = None
  critic_encoder_configs: dict[str, Any] | str | None = None
  vae_encoder_kwargs: dict[str, Any] | None = None
  vae_decoder_kwargs: dict[str, Any] | None = None
  vae_latent_size: int | None = None
  vae_input_subobs_components: tuple[str, ...] | None = None
  vae_aux_subobs_components: tuple[str, ...] | None = None


@dataclass
class InstinctRlPpoAlgorithmCfg:
  class_name: str = "PPO"
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  entropy_coef: float = 0.005
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1.0e-3
  optimizer_class_name: str = "AdamW"
  schedule: str = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  advantage_mixing_weights: float | tuple[float, ...] = 1.0
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  clip_min_std: float = 1.0e-12
  kl_loss_func: str = "kl_divergence"
  kl_loss_coef: float = 1.0
  using_ppo: bool = True
  teacher_policy_class_name: str = "ActorCritic"
  teacher_policy: dict[str, Any] = field(default_factory=dict)
  teacher_logdir: str | None = None
  label_action_with_critic_obs: bool = True
  teacher_act_prob: str | float = "exp"
  update_times_scale: int = 100
  distillation_loss_coef: float | str = 1.0
  distill_target: str = "real"
  actor_state_key: str = "amp_policy"
  reference_state_key: str = "amp_reference"
  discriminator_class_name: str = "Discriminator"
  discriminator_kwargs: dict[str, Any] = field(default_factory=dict)
  discriminator_optimizer_class_name: str = "AdamW"
  discriminator_optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
  discriminator_reward_coef: float = 1.0
  discriminator_reward_type: str = "log"
  discriminator_loss_func: str = "BCEWithLogitsLoss"
  discriminator_loss_coef: float = 1.0
  discriminator_gradient_penalty_coef: float = 10.0
  discriminator_weight_decay_coef: float = 0.0
  discriminator_logit_weight_decay_coef: float = 0.0
  discriminator_gradient_torlerance: float = 0.0
  discriminator_backbone_gradient_only: bool = False


@dataclass
class InstinctRlNormalizerCfg:
  class_name: str = "EmpiricalNormalization"


@dataclass
class InstinctRlOnPolicyRunnerCfg:
  seed: int = 42
  device: str = "cuda:0"
  num_steps_per_env: int = 24
  max_iterations: int = 30_000
  policy: InstinctRlActorCriticCfg = field(default_factory=InstinctRlActorCriticCfg)
  algorithm: InstinctRlPpoAlgorithmCfg = field(default_factory=InstinctRlPpoAlgorithmCfg)
  normalizers: dict[str, InstinctRlNormalizerCfg] = field(
    default_factory=lambda: {
      "policy": InstinctRlNormalizerCfg(),
      "critic": InstinctRlNormalizerCfg(),
    }
  )
  save_interval: int = 500
  log_interval: int = 1
  experiment_name: str = "instinct_mjlab"
  run_name: str = ""
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"
  ckpt_manipulator: str | None = None
  ckpt_manipulator_kwargs: dict[str, Any] = field(default_factory=dict)
  policy_observation_group: str = "actor"
  critic_observation_group: str = "critic"

  def to_dict(self) -> dict[str, Any]:
    return _to_plain_dict(self)
