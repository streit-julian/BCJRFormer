import logging
import os
from typing import cast

import torch
from dotenv import load_dotenv
from spock import SpockBuilder
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LRScheduler,
    SequentialLR,
)
from torch.utils.data import DataLoader

import wandb
from bcjrformer.configs.channel_config import IDSChannelConfig
from bcjrformer.configs.code_config import (
    ConcatenatedCodeConfig,
    ConvolutionalCodeConfig,
    InnerCodeConfig,
    LinearCodeConfig,
    MarkerCodeConfig,
)
from bcjrformer.configs.inner_model_config import InnerModelConfig
from bcjrformer.configs.model_config import ModelConfig
from bcjrformer.configs.specific_model_config import (
    BCJRFormerModel,
    CombinedConvBCJRFormerModel,
    ConvBCJRFormerModel,
    EcctOnBCJRFormerE2EModel,
    IdsECCTOuterModel,
    SpecificModelConfig,
    specific_model_instance_to_identifier,
    specific_model_to_wandb_project,
)
from bcjrformer.configs.build_run_name_util import build_run_name_and_inner_config
from bcjrformer.configs.utils import SchedulerType
from bcjrformer.model_builder import model_builder_factory
from bcjrformer.utils import (
    code_exclude_fn,
    get_ssh_server_name,
    set_seed,
    setup_cuda_device,
    setup_logger,
)

load_dotenv()

TRAIN_SPOCK_CONFIGS = (
    ModelConfig,
    ConcatenatedCodeConfig,
    LinearCodeConfig,
    MarkerCodeConfig,
    ConvolutionalCodeConfig,
    IDSChannelConfig,
    InnerModelConfig,
    BCJRFormerModel,
    CombinedConvBCJRFormerModel,
    ConvBCJRFormerModel,
    EcctOnBCJRFormerE2EModel,
    IdsECCTOuterModel,
)


def create_scheduler(
    model_config: ModelConfig, optimizer: torch.optim.Optimizer
) -> LRScheduler:

    if model_config.scheduler_type == SchedulerType.COSINE:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=model_config.epochs - model_config.warmup_epochs,
            eta_min=model_config.lr / 10,
        )
    else:
        # compare with constant learning rater (i. e. we don't change the learning rate)
        scheduler = ConstantLR(optimizer, factor=1.0)

    if model_config.warmup_epochs > 0:
        schedulers: list[LRScheduler] = [
            ConstantLR(optimizer, factor=1.0, total_iters=model_config.warmup_epochs),
            scheduler,
        ]

        scheduler = SequentialLR(
            optimizer,
            schedulers=schedulers,
            milestones=[model_config.warmup_epochs],
        )

    return scheduler


def train_model(
    model_config: ModelConfig,
    specific_model_config: SpecificModelConfig,
    channel_config: IDSChannelConfig,
    model_dir: str,
    logger: logging.Logger,
    outer_code_config: LinearCodeConfig,
    inner_code_config: InnerCodeConfig | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")

    logger.debug("Getting datasets and trainer")

    builder = model_builder_factory(specific_model_config)

    train_dataset, test_dataset, trainer = builder.build_for_train(
        model_config,
        channel_config,
        outer_code_config,
        inner_code_config,
        device,
        logger,
    )

    logger.debug("Initializing dataloaders")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=model_config.workers,
        worker_init_fn=(
            train_dataset.worker_init_fn  # type: ignore
            if hasattr(train_dataset, "worker_init_fn")
            else None
        ),
        prefetch_factor=(
            2 * model_config.batch_accumulation if model_config.workers > 0 else None
        ),
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=model_config.test_batch_size,
        shuffle=False,
        num_workers=model_config.workers,
    )

    logger.debug("Initializing optimizer and scheduler")

    #################################
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=model_config.lr)

    scheduler: LRScheduler = create_scheduler(model_config, optimizer)

    logger.info("Starting training..")
    trainer.train(model_dir, optimizer, scheduler, train_dataloader)

    logger.info("Starting evaluation..")
    trainer.evaluate(test_dataloader)

    logger.info("Finished training and evaluation")


def parse_args() -> tuple[
    ModelConfig,
    ConcatenatedCodeConfig,
    IDSChannelConfig,
]:
    config = SpockBuilder(
        *TRAIN_SPOCK_CONFIGS,
        description="BCJRFormer for transmissions via the IDS-Channel",
    ).generate()

    model_config: ModelConfig = config.ModelConfig

    channel_config: IDSChannelConfig = cast(IDSChannelConfig, model_config.ids_channel)

    code_config: ConcatenatedCodeConfig = cast(
        ConcatenatedCodeConfig, config.ConcatenatedCodeConfig
    )

    return model_config, code_config, channel_config


def initialize_model_path(
    model_config: ModelConfig, specific_model_identifier: str, run_name: str
):
    if model_config.model_base_dir is not None:
        model_dir = os.path.join(
            model_config.model_base_dir,
            f"Results_{specific_model_identifier}",
            run_name,
        )
    else:
        model_dir = os.path.join(
            f"Results_{specific_model_identifier}",
            run_name,
        )

    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def main():
    model_config, code_config, channel_config = parse_args()

    set_seed(model_config.seed)
    ####################################################################

    outer_code_config = code_config.outer_code_config

    specific_model_config = cast(SpecificModelConfig, model_config.specific_model)
    specific_model_identifier = specific_model_instance_to_identifier(
        specific_model_config
    ).value
    run_name, inner_code_wandb_config = build_run_name_and_inner_config(
        specific_model_config, code_config, channel_config
    )
    model_dir = initialize_model_path(model_config, specific_model_identifier, run_name)

    # Setup logging
    logger = setup_logger(
        __name__,
        extra_handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(model_dir, "train.log")),
        ],
    )

    # Setup CUDA Devices
    setup_cuda_device(model_config.gpu)
    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
        torch.set_float32_matmul_precision("high")
        logger.info("Using tfloat32 high precision matmul")

    if model_config.custom_run_suffix:
        suffix = model_config.custom_run_suffix
        run_name += f"__{suffix}"

    # Initialize Wandb
    if model_config.log_wandb:
        project = specific_model_to_wandb_project(specific_model_config).value
        wandb.init(project=project, name=run_name)

        if model_config.wandb_include_code:
            wandb.run.log_code(".", exclude_fn=code_exclude_fn)  # type: ignore

        wandb.config.update(model_config)  # type: ignore
        wandb.config.update(channel_config)  # type: ignore
        wandb.config.update(outer_code_config)  # type: ignore
        wandb.config.update(inner_code_wandb_config)

        # add ssh server so we know where the files are stored
        wandb.config.update(
            {
                "specific_model": specific_model_identifier,
                "ssh_server_name": get_ssh_server_name(logger),
            },
            allow_val_change=True,
        )  # type: ignore

    ####################################################################

    logger.info(f"Selected model is {model_config.specific_model}")
    logger.info("Config: %s", model_config)
    logger.info("Linear Code Config: %s", outer_code_config)
    logger.info("Inner Code Config: %s", code_config.inner_code_config)
    logger.info("Channel Config: %s", channel_config)
    logger.info(f"Path to model/logs: {model_dir}")

    # Train the model
    train_model(
        model_config,
        specific_model_config,
        channel_config,
        model_dir,
        logger,
        code_config.outer_code_config,
        code_config.inner_code_config,
    )

    if model_config.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
