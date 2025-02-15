from spock import spock
from bcjrformer.configs.channel_config import IDSChannelConfig
from bcjrformer.configs.inner_model_config import InnerModelConfig
from bcjrformer.configs.specific_model_config import SpecificModel
from bcjrformer.configs.utils import BooleanFlag, SchedulerType


@spock
class EvaluationConfig:
    model_path: str

    specific_model: SpecificModel = SpecificModel.BCJRFORMER

    ids_channel: IDSChannelConfig
    inner_model_config: InnerModelConfig

    # while in the context of an iteratively generated dataset meaningless parameter
    # it is meaningful for the update of the scheduler and logging
    masked_attention: BooleanFlag = BooleanFlag.TRUE
    workers: int = 4
    lr: float = 1e-4
    scheduler_type: SchedulerType = SchedulerType.CONSTANT
    gpu: int = -1

    test_batch_size: int = 128
    test_batches_per_epoch: int = 1000

    compile_model: BooleanFlag = BooleanFlag.FALSE

    N_dec: int = 6
    d_model: int = 32
    h: int = 8

    seed: int = 42
    log_wandb: BooleanFlag = BooleanFlag.FALSE

    custom_run_suffix: str | None = None
