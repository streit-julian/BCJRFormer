import os
from spock import spock
from spock.utils import ge
from bcjrformer.configs.channel_config import IDSChannelConfig
from bcjrformer.configs.inner_model_config import InnerModelConfig
from bcjrformer.configs.specific_model_config import SpecificModel
from bcjrformer.configs.utils import BooleanFlag, SchedulerType


@spock
class ModelConfig:

    specific_model: SpecificModel = SpecificModel.BCJRFORMER

    ids_channel: IDSChannelConfig
    inner_model_config: InnerModelConfig

    # while in the context of an iteratively generated dataset meaningless parameter
    # it is meaningful for the update of the scheduler and logging
    masked_attention: BooleanFlag = BooleanFlag.TRUE
    epochs: int = 1000
    workers: int = 4
    lr: float = 1e-4
    scheduler_type: SchedulerType = SchedulerType.CONSTANT
    dropout: float = 0.0
    gpu: int = -1
    batch_size: int = 128
    batch_accumulation: int = 1
    batches_per_epoch: int = 1000

    compile_model: BooleanFlag = BooleanFlag.FALSE

    warmup_epochs: int = 0

    # Test batch size
    test_batch_size: int = 2048
    test_batches_per_epoch: int = 1000

    seed: int = 42
    N_dec: int = 6
    d_model: int = 32
    h: int = 8

    log_wandb: BooleanFlag = BooleanFlag.FALSE
    wandb_include_code: BooleanFlag = BooleanFlag.FALSE

    save_checkpoints_every_n_epochs: int | None = None

    # whether to train from checkpoint file
    from_checkpoint: BooleanFlag = BooleanFlag.FALSE
    # base directory of the checkpoint to load
    checkpoint_dir: str | None = None
    # which epoch - checkpoint - If none, continue training from best-model
    from_checkpoint_epoch: int | None = None

    custom_run_suffix: str | None = None

    # Base Directory where the training results will be saved to, i. e. checkpoints + log
    model_base_dir: str | None = None

    def __post_hook__(self):
        if self.from_checkpoint and self.checkpoint_dir is None:
            raise ValueError(
                "Checkpoint directory must be provided if 'from_checkpoint' is True"
            )

        if self.save_checkpoints_every_n_epochs is not None:
            ge(self.save_checkpoints_every_n_epochs, 1)

        if self.from_checkpoint_epoch is not None:
            ge(self.from_checkpoint_epoch, 1)

        if self.model_base_dir is not None and not os.path.exists(self.model_base_dir):
            raise FileNotFoundError(
                f"Model base directory {self.model_base_dir} does not exist"
            )
