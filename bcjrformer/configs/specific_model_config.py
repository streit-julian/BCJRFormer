from enum import Enum, StrEnum
from spock import spock


@spock
class IdsECCTOuterModel:
    pass


@spock
class EcctOnBCJRFormerE2EModel:
    inner_pre_trained_model_path: str | None = None


@spock
class BCJRFormerModel:
    pass


@spock
class CombinedConvBCJRFormerModel:
    n_masked_heads: int = 0


@spock
class ConvBCJRFormerModel:
    N_dec_symbol: int = 1
    N_dec_state: int = 1
    N_dec_cross: int = 1
    h_symbol: int = 8
    h_state: int = 8
    h_symbol_to_state: int = 8
    h_state_to_symbol: int = 8


class SpecificModel(Enum):
    BCJRFORMER = BCJRFormerModel
    IDS_ECCT_OUTER = IdsECCTOuterModel
    ECCT_ON_BCJRFORMER_SEP_E2E = EcctOnBCJRFormerE2EModel
    COMBINED_CONV_BCJRFORMER = CombinedConvBCJRFormerModel
    CONV_BCJRFORMER = ConvBCJRFormerModel


SpecificModelConfig = (
    BCJRFormerModel
    | IdsECCTOuterModel
    | EcctOnBCJRFormerE2EModel
    | CombinedConvBCJRFormerModel
    | ConvBCJRFormerModel
)


class SpecificModelIdentifier(StrEnum):
    BCJRFORMER = "BCJRFORMER"
    IDS_ECCT_OUTER = "IDS_ECCT_OUTER"
    ECCT_ON_BCJRFORMER_SEP_E2E = "ECCT_ON_BCJRFORMER_E2E"
    COMBINED_CONV_BCJRFORMER = "COMBINED_CONV_BCJRFORMER"
    CONV_BCJRFORMER = "CONV_BCJRFORMER"


class WandbProjectNames(StrEnum):
    OUTER_DECODING = "OUTER_DECODING"
    INNER_DECODING = "INNER_DECODING"


def specific_model_instance_to_identifier(
    model_config: SpecificModelConfig,
) -> SpecificModelIdentifier:
    if isinstance(model_config, BCJRFormerModel):
        return SpecificModelIdentifier.BCJRFORMER
    if isinstance(model_config, IdsECCTOuterModel):
        return SpecificModelIdentifier.IDS_ECCT_OUTER
    if isinstance(model_config, EcctOnBCJRFormerE2EModel):
        return SpecificModelIdentifier.ECCT_ON_BCJRFORMER_SEP_E2E
    if isinstance(model_config, CombinedConvBCJRFormerModel):
        return SpecificModelIdentifier.COMBINED_CONV_BCJRFORMER
    if isinstance(model_config, ConvBCJRFormerModel):
        return SpecificModelIdentifier.CONV_BCJRFORMER
    raise ValueError(f"Unknown model config: {model_config}")


def specific_model_to_wandb_project(
    model_config: SpecificModelConfig,
) -> WandbProjectNames:
    if isinstance(model_config, BCJRFormerModel):
        return WandbProjectNames.INNER_DECODING
    if isinstance(model_config, IdsECCTOuterModel):
        return WandbProjectNames.OUTER_DECODING
    if isinstance(model_config, EcctOnBCJRFormerE2EModel):
        return WandbProjectNames.OUTER_DECODING
    if isinstance(model_config, CombinedConvBCJRFormerModel):
        return WandbProjectNames.INNER_DECODING
    if isinstance(model_config, ConvBCJRFormerModel):
        return WandbProjectNames.INNER_DECODING
    raise ValueError(f"Unknown model config: {model_config}")
