# coding=utf-8
"""Configuration classes for Probabilistic Transformer CLM experiments."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class PtConfig(PretrainedConfig):
    """Configuration for PT-style iterative dependency inference.

    The base ``pt`` model is the softmax dependency-head baseline. Variant
    subclasses below only change default normalization/gating choices so they
    can be addressed through Hugging Face ``model_type`` names.
    """

    model_type = "pt"
    default_head_selection_method = "softmax"
    default_channel_gate_type = "none"
    default_pairwise_channel_softmax = False

    def __init__(
        self,
        vocab_size=32000,
        dim_z=4096,
        dim_g=11008,
        num_iterations=32,
        num_channels=32,
        ternary_rank=None,
        potential_func_z="square",
        potential_func_g="abs",
        max_position_embeddings=2048,
        initializer_range=0.02,
        binary_initializer_range=0.02,
        ternary_initializer_range=0.02,
        binary_factor_scaling=1.0,
        ternary_factor_scaling=1.0,
        classifier_amplifier=1.0,
        potential_eps=1e-6,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        dropout_prob_z=0.1,
        dropout_prob_h=0.1,
        classifier_dropout=None,
        regularize_z=1.0,
        regularize_h=1.0,
        regularize_g=1.0,
        is_causal=True,
        mask_self=False,
        head_selection_method=None,
        channel_gate_type=None,
        pairwise_channel_softmax=None,
        channel_gate_temperature=1.0,
        normalized_sigmoid_eps=1e-6,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.dim_z = dim_z
        self.dim_g = dim_g

        self.num_iterations = num_iterations
        self.num_channels = num_channels
        if ternary_rank is None:
            ternary_rank = dim_z // num_channels
        self.ternary_rank = ternary_rank

        self.potential_func_z = potential_func_z
        self.potential_func_g = potential_func_g
        self.initializer_range = initializer_range
        self.binary_initializer_range = binary_initializer_range
        self.ternary_initializer_range = ternary_initializer_range
        self.binary_factor_scaling = binary_factor_scaling
        self.ternary_factor_scaling = ternary_factor_scaling
        self.classifier_amplifier = classifier_amplifier
        self.potential_eps = potential_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.dropout_prob_z = dropout_prob_z
        self.dropout_prob_h = dropout_prob_h
        self.classifier_dropout = classifier_dropout
        self.regularize_z = regularize_z
        self.regularize_h = regularize_h
        self.regularize_g = regularize_g

        self.is_causal = is_causal
        self.mask_self = mask_self
        self.head_selection_method = head_selection_method or self.default_head_selection_method
        self.channel_gate_type = channel_gate_type or self.default_channel_gate_type
        if pairwise_channel_softmax is None:
            pairwise_channel_softmax = self.default_pairwise_channel_softmax
        self.pairwise_channel_softmax = pairwise_channel_softmax
        self.channel_gate_temperature = channel_gate_temperature
        self.normalized_sigmoid_eps = normalized_sigmoid_eps

        # Fields expected by common HF heads/utilities.
        self.hidden_size = dim_z
        self.hidden_act = kwargs.pop("hidden_act", "silu")
        self.layer_norm_eps = kwargs.pop("layer_norm_eps", 1e-5)

        self._validate_variant_options()
        self._rope_scaling_validation()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _validate_variant_options(self):
        valid_head_methods = {
            "softmax",
            "sel_softmax",
            "sigmoid",
            "sigmoid_with_b",
            "normalized_sigmoid",
            "linear",
        }
        valid_gate_types = {"none", "static", "token", "message"}
        if self.head_selection_method not in valid_head_methods:
            raise ValueError(
                f"Unknown PT head_selection_method={self.head_selection_method!r}. "
                f"Expected one of {sorted(valid_head_methods)}."
            )
        if self.channel_gate_type not in valid_gate_types:
            raise ValueError(
                f"Unknown PT channel_gate_type={self.channel_gate_type!r}. "
                f"Expected one of {sorted(valid_gate_types)}."
            )
        if self.channel_gate_temperature <= 0:
            raise ValueError("channel_gate_temperature must be positive.")
        if self.normalized_sigmoid_eps <= 0:
            raise ValueError("normalized_sigmoid_eps must be positive.")

    def _rope_scaling_validation(self):
        if self.rope_scaling is None:
            return
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling.type` must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling.factor` must be a float > 1, got {rope_scaling_factor}")


class PtSelSoftmaxConfig(PtConfig):
    model_type = "pt-sel-softmax"
    default_head_selection_method = "sel_softmax"


class PtSigmoidConfig(PtConfig):
    model_type = "pt-sigmoid"
    default_head_selection_method = "sigmoid"


class PtSigmoidWithBConfig(PtConfig):
    model_type = "pt-sigmoid-with-b"
    default_head_selection_method = "sigmoid_with_b"


class PtNormalizedSigmoidConfig(PtConfig):
    model_type = "pt-normalized-sigmoid"
    default_head_selection_method = "normalized_sigmoid"


class PtLinearConfig(PtConfig):
    model_type = "pt-linear"
    default_head_selection_method = "linear"


class PtChannelGateConfig(PtConfig):
    model_type = "pt-channel-gate"
    default_head_selection_method = "softmax"
    default_channel_gate_type = "token"


class PtSigmoidChannelGateConfig(PtConfig):
    model_type = "pt-sigmoid-channel-gate"
    default_head_selection_method = "sigmoid_with_b"
    default_channel_gate_type = "token"


class PtHeadSoftmaxConfig(PtConfig):
    model_type = "pt-head-softmax"
    default_head_selection_method = "softmax"
    default_channel_gate_type = "static"


class PtHeadSoftmaxWithBConfig(PtConfig):
    model_type = "pt-head-softmax-with-b"
    default_head_selection_method = "sigmoid_with_b"
    default_pairwise_channel_softmax = True


class PtSoftmaxAndHeadSoftmaxConfig(PtConfig):
    model_type = "pt-softmax-and-head-softmax"
    default_head_selection_method = "softmax"
    default_pairwise_channel_softmax = True
