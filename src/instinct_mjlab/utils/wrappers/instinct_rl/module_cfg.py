from dataclasses import dataclass, field
from typing import List, Literal, Sequence

@dataclass(kw_only=True)
class InstinctRlParallelBlockCfg:
    """Configuration for the encoder network."""

    class_name: str = None
    """The encoder class name. Default is Encoder."""

    component_names: List[str] = None
    """The names of the observation components to be embedded."""

    output_size: int = None
    """The output size of the encoder network."""

    takeout_input_components: bool = True
    """Whether to take out the input components from the embedded obs to the rest of the network."""

@dataclass(kw_only=True)
class InstinctRlMlpCfg(InstinctRlParallelBlockCfg):
    """Configuration for the MLP encoder network."""

    class_name: str = "MlpModel"
    """The encoder class name. Default is MlpModel."""

    hidden_sizes: List[int] = None
    """The hidden dimensions of the encoder network."""

    nonlinearity: str = None
    """The activation function for the encoder network."""

@dataclass(kw_only=True)
class InstinctRlConv2dHeadCfg(InstinctRlParallelBlockCfg):
    """Configuration for the Conv2d encoder network."""

    class_name: str = "Conv2dHeadModel"
    """The encoder class name. Default is Conv2dHeadModel."""

    channels: List[int] = None
    """The number of channels."""

    kernel_sizes: List[int] = None
    """The size of the kernel."""

    strides: List[int] = None
    """The stride of the kernel."""

    hidden_sizes: List[int] = None
    """The hidden dimensions of the output mlp head."""

    paddings: List[int] = None
    """The padding of the kernel."""

    nonlinearity: str = None
    """The activation function for the encoder network."""

    use_maxpool: bool = False
    """Whether to use max pooling in the convolutional layers."""

@dataclass(kw_only=True)
class InstinctRlTransformerHeadCfg(InstinctRlParallelBlockCfg):
    """Configuration for the Transformer encoder network."""

    class_name: str = "TransformerHeadModel"
    """The class name. Default is TransformerHeadModel."""

    num_heads: int = 4
    """The number of attention heads."""

    num_layers: int = 1
    """The number of transformer encoder layers."""

    d_model: int = 256
    """The latent size of the transformer encoder."""

    dim_feedforward: int = 512
    """The feedforward dimension of the transformer encoder. Default in Transformer is 2048, we use 512."""

    dropout: float = 0.1
    """The dropout rate."""

    activation: str = "relu"
    """The activation function for the transformer encoder."""

    nonlinearity: str = "ReLU"
    """The nonlinearity layer for the mlp network."""

    layer_norm_eps: float = 1e-5
    """The epsilon value for layer normalization."""

    batch_first: bool = True
    """Whether the input is batch first."""

    norm_first: bool = False
    """Whether to apply normalization first."""

    mask_from_input_dim: int = -1
    """The dimension to get the self-attention mask from the input tensor. If -1, no mask is used."""

    output_selection: Literal["maxpool", "smallest_positive", "smallest_nonnegative"] = "maxpool"
    """The output selection method."""

    input_hidden_sizes: List[int] = field(default_factory=list)
    """The hidden dimensions of the input mlp head. If None, only a linear layer is used."""

    output_hidden_sizes: List[int] = field(default_factory=list)
    """The hidden dimensions of the output mlp head. If None, only a linear layer is used."""
