from datetime import datetime
from enum import IntEnum, StrEnum

from bcjrformer.configs.channel_config import IDSChannelConfig
from bcjrformer.configs.code_config import (
    ConcatenatedCodeConfig,
    ConvolutionalCodeConfig,
    MarkerCodeConfig,
    code_config_to_code_type,
)
from bcjrformer.configs.specific_model_config import (
    SpecificModelConfig,
    specific_model_instance_to_identifier,
)
from typing import cast


def build_run_name_and_inner_config(
    specific_model_config: SpecificModelConfig,
    code_config: ConcatenatedCodeConfig,
    channel_config: IDSChannelConfig,
) -> tuple[str, dict]:
    """
    Build the run name and inner configuration for logging.

    Args:
        model_config (ModelConfig): The model configuration.
        code_config (ConcatenatedCodeConfig): The code configuration.
        channel_config (): The channel configuration.
        linear_code (LinearCode): The linear code.

    Returns:
        tuple[str, dict]: The run name and inner configuration dictionary.
    """
    run_name = (
        f"{datetime.now().strftime('%d%m%Y_%H%M%S')}_"
        f"{specific_model_instance_to_identifier(specific_model_config).value}_"
        f"channel_pi_{channel_config.p_i}_pd_{channel_config.p_d}_ps_{channel_config.p_s}_"
        f"{code_config.outer_code_config.code_type}_"
        f"n_{code_config.outer_code_config.n}_"
        f"k_{code_config.outer_code_config.k}"
    )

    inner_code_wandb_config: dict = {}
    if code_config.inner_code_config is not None:
        inner_code_config = cast(
            ConvolutionalCodeConfig | MarkerCodeConfig, code_config.inner_code_config
        )

        inner_code_type = code_config_to_code_type(inner_code_config)
        inner_code_wandb_config["inner_code_type"] = inner_code_type.value

        if isinstance(inner_code_config, ConvolutionalCodeConfig):
            run_name += (
                f"_CONV_k_{inner_code_config.k}_"
                f"g_{inner_code_config.g}_"
                f"p_{inner_code_config.p}"
            )

            inner_code_wandb_config["conv_k"] = inner_code_config.k
            inner_code_wandb_config["conv_g"] = inner_code_config.g
            inner_code_wandb_config["conv_p"] = inner_code_config.p

        elif isinstance(inner_code_config, MarkerCodeConfig):
            run_name += (
                f"_MARKER_{inner_code_config.marker}_"
                f"Nc_{inner_code_config.N_c}_"
                f"p_{inner_code_config.p}"
            )

            inner_code_wandb_config["marker"] = inner_code_config.marker
            inner_code_wandb_config["marker_Nc"] = inner_code_config.N_c
            inner_code_wandb_config["marker_p"] = inner_code_config.p

        else:
            raise ValueError("Unknown inner code config")
    return run_name, inner_code_wandb_config
