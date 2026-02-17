from dataclasses import dataclass, field
from typing import Literal, Sequence

from instinct_mjlab.utils.wrappers.instinct_rl.module_cfg import InstinctRlParallelBlockCfg

@dataclass(kw_only=True)
class InstinctRlActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = None
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = None
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = None
    """The hidden dimensions of the critic network."""

    activation: str = None
    """The activation function for the actor and critic networks."""

@dataclass(kw_only=True)
class InstinctRlActorCriticRecurrentCfg(InstinctRlActorCriticCfg):
    class_name: str = "ActorCriticRecurrent"

    rnn_type: str = "gru"
    """The type of RNN to use. Default is GRU."""

    rnn_hidden_size: int = 256
    """The hidden size of the RNN."""

    rnn_num_layers: int = 1
    """The number of layers in the RNN."""

    multireward_multirnn: bool = False
    """Whether to use multiple RNN critics for multiple rewards."""

@dataclass(kw_only=True)
class EncoderCfgMixin:
    encoder_configs: object = None
    """ A dataclass containing InstinctRlEncoderCfg.
    """

    critic_encoder_configs: object | Literal["shared"] | None = None
    """ A dataclass containing InstinctRlEncoderCfg for building the critic encoders.
    """

@dataclass(kw_only=True)
class EstimatorCfgMixin:
    """Mixin for state estimator."""

    estimator_obs_components: list[str] = None
    """The components of the observation used for the estimator."""

    estimator_target_components: list[str] = None
    """The components of the observation used as the target for the estimator."""

    estimator_configs: object = None
    """A dataclass containing GalaxeaRlEncoderCfg for building the estimator."""

    replace_state_prob: float = 1.0

@dataclass(kw_only=True)
class EstimatorActorCriticCfg(
    EstimatorCfgMixin,
    InstinctRlActorCriticCfg,
):
    """Configuration for the estimator actor-critic networks."""

    class_name: str = "EstimatorActorCritic"


@dataclass(kw_only=True)
class EstimatorActorCriticRecurrentCfg(
    EstimatorCfgMixin,
    InstinctRlActorCriticRecurrentCfg,
):
    """Configuration for the estimator actor-critic-recurrent networks."""

    class_name: str = "EstimatorActorCriticRecurrent"


@dataclass(kw_only=True)
class InstinctRlMoEActorCriticCfg(InstinctRlActorCriticCfg):
    class_name: str = "MoEActorCritic"

    num_moe_experts: int = 8
    """The number of Mixture of Experts (MoE) experts."""

    moe_gate_hidden_dims: list[int] = field(default_factory=list)
    """The hidden dimensions of the MoE gate network."""

@dataclass(kw_only=True)
class InstinctRlVaeActorCriticCfg(InstinctRlActorCriticCfg):
    class_name: str = "VaeActor"

    vae_encoder_kwargs: dict = None
    """ A dict building the MLP-based VAE encoder."""

    vae_decoder_kwargs: dict = None
    """ A dict building the MLP-based VAE decoder."""

    vae_latent_size: int = 16
    """ The latent size of the VAE."""

    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    """The hidden dimensions of the critic network (typically not used for VAE distillation)."""

    init_noise_std: float = 1e-4
    """The initial noise standard deviation for the critic network (typically not used for VAE distillation)."""

    activation: str = "elu"
    """The activation function for the critic network (typically not used for VAE distillation)."""

@dataclass(kw_only=True)
class InstinctRlEncoderActorCriticCfg(
    EncoderCfgMixin,
    InstinctRlActorCriticCfg,
):
    """Configuration for the encoder actor-critic networks."""

    class_name: str = "EncoderActorCritic"


@dataclass(kw_only=True)
class InstinctRlEncoderActorCriticRecurrentCfg(
    EncoderCfgMixin,
    InstinctRlActorCriticRecurrentCfg,
):
    """Configuration for the encoder actor-critic-recurrent networks."""

    class_name: str = "EncoderActorCriticRecurrent"


@dataclass(kw_only=True)
class InstinctRlEncoderMoEActorCriticCfg(
    EncoderCfgMixin,
    InstinctRlMoEActorCriticCfg,
):
    """Configuration for the encoder actor-critic networks."""

    class_name: str = "EncoderMoEActorCritic"


@dataclass(kw_only=True)
class InstinctRlEncoderVaeActorCriticCfg(
    EncoderCfgMixin,
    InstinctRlVaeActorCriticCfg,
):
    """Configuration for the encoder actor networks."""

    class_name: str = "EncoderVaeActorCritic"


@dataclass(kw_only=True)
class InstinctRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = None
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = None
    """Whether to use clipped value loss."""

    clip_param: float = None
    """The clipping parameter for the policy."""

    entropy_coef: float = None
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = None
    """The number of learning epochs per update."""

    num_mini_batches: int = None
    """The number of mini-batches per update."""

    learning_rate: float = None
    """The learning rate for the policy."""

    optimizer_class_name: str = "AdamW"
    """The optimizer class name. Default is AdamW."""

    schedule: str = None
    """The learning rate schedule."""

    gamma: float = None
    """The discount factor."""

    lam: float = None
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    advantage_mixing_weights: float | Sequence[float] = 1.0
    """The weights for the mixing advantages and compute surrogate loss when multiple rewards are returned."""

    desired_kl: float = None
    """The desired KL divergence."""

    max_grad_norm: float = None
    """The maximum gradient norm."""

    clip_min_std: float = 1e-12
    """The minimum standard deviation for the policy when computing distribution.
    Default: 1e-12 to prevent numerical instability.
    """

@dataclass(kw_only=True)
class InstinctRlNormalizerCfg:
    class_name: str = "EmpiricalNormalization"

@dataclass(kw_only=True)
class InstinctRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = None
    """The number of steps per environment per update."""

    max_iterations: int = None
    """The maximum number of iterations."""

    ckpt_manipulator: str | None = None
    """A string calling the checkpoint manipulator when loading. Typically, user has to implement their own
    checkpoint manipulator to load the model weights if the loaded model is different from the current training
    model. Default is None for no manipulation.
    """

    policy: InstinctRlActorCriticCfg = None
    """The policy configuration."""

    algorithm: InstinctRlPpoAlgorithmCfg = None
    """The algorithm configuration."""

    normalizers: dict = field(default_factory=dict)
    """The configs for each observation group when they are still flattened tensors
    Empty dict for no normalizer running in the RL runner.
    """

    ##
    # Checkpointing parameters
    ##

    save_interval: int = None
    """The number of iterations between saves."""

    log_interval: int = 1
    """The number of iterations between logs."""

    experiment_name: str = None
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    # logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    # """The logger to use. Default is tensorboard."""

    # neptune_project: str = "instinctlab"
    # """The neptune project name. Default is "instinctlab"."""

    # wandb_project: str = "instinctlab"
    # """The wandb project name. Default is "instinctlab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str | None = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    policy_observation_group: str = "policy"
    """The observation group name for the policy network. Default is "policy"."""

    critic_observation_group: str = "critic"
    """The observation group name for the critic network. Default is "critic"."""
