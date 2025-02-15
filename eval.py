import logging
import os
from typing import cast

import torch
from dotenv import load_dotenv
from spock import SpockBuilder
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
from bcjrformer.configs.evaluation_config import EvaluationConfig
from bcjrformer.configs.inner_model_config import InnerModelConfig
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
from bcjrformer.model_builder import model_builder_factory
from bcjrformer.utils import (
    get_ssh_server_name,
    repair_compiled_state_dict,
    set_seed,
    setup_cuda_device,
    setup_logger,
)

load_dotenv()

EVAL_SPOCK_CONFIGS = (
    EvaluationConfig,
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


def evaluate_model(
    eval_cfg: EvaluationConfig,
    specific_model_config: SpecificModelConfig,
    channel_config: IDSChannelConfig,
    logger: logging.Logger,
    outer_code_config: LinearCodeConfig,
    inner_code_config: InnerCodeConfig | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.debug("Getting datasets and trainer for evaluation")

    builder = model_builder_factory(specific_model_config)

    _, test_dataset, trainer = builder.build_for_evaluation(
        eval_cfg,
        channel_config,
        outer_code_config,
        inner_code_config,
        device,
        logger,
    )

    logger.debug(f"Loading model from checkpoint: {eval_cfg.model_path}")
    checkpoint_dict = torch.load(eval_cfg.model_path)

    model_config = checkpoint_dict["config"]
    model_state_dict = checkpoint_dict["model_state_dict"]

    if model_config.compile_model and not eval_cfg.compile_model:
        logger.info("Reparsing model config to compile model")
        model_state_dict = repair_compiled_state_dict(model_state_dict)

    trainer.model.load_state_dict(model_state_dict)

    logger.debug("Initializing test dataloader")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=eval_cfg.test_batch_size,
        shuffle=False,
        num_workers=eval_cfg.workers,
    )

    logger.info("Starting evaluation...")
    trainer.evaluate(test_dataloader)
    logger.info("Finished evaluation")


def parse_args() -> tuple[
    EvaluationConfig,
    ConcatenatedCodeConfig,
    IDSChannelConfig,
]:
    config = SpockBuilder(
        *EVAL_SPOCK_CONFIGS,
        description="BCJRFormer Evaluation via the IDS-Channel",
    ).generate()

    eval_config: EvaluationConfig = config.EvaluationConfig
    channel_config: IDSChannelConfig = cast(IDSChannelConfig, eval_config.ids_channel)
    code_config: ConcatenatedCodeConfig = cast(
        ConcatenatedCodeConfig, config.ConcatenatedCodeConfig
    )
    return eval_config, code_config, channel_config


def main():
    # Parse the evaluation configuration along with the code config
    eval_config, code_config, channel_config = parse_args()

    # Use the provided seed (or default to 42) for reproducibility.

    set_seed(eval_config.seed)

    # Build the linear code based on the outer code config.
    outer_code_config = code_config.outer_code_config

    specific_model_config = cast(SpecificModelConfig, eval_config.specific_model)
    specific_model_identifier = specific_model_instance_to_identifier(
        specific_model_config
    ).value
    run_name, inner_code_wandb_config = build_run_name_and_inner_config(
        specific_model_config, code_config, channel_config
    )

    eval_log_dir = os.path.join(os.path.dirname(eval_config.model_path), "eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)
    logger = setup_logger(
        __name__,
        extra_handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(eval_log_dir, "eval.log")),
        ],
    )

    logger.info("Selected model: %s", eval_config.specific_model)
    logger.info("Evaluation Config: %s", eval_config)
    logger.info("Linear Code Config: %s", outer_code_config)
    logger.info("Channel Config: %s", channel_config)
    logger.info("Loading model from path: %s", eval_config.model_path)

    # Setup CUDA device
    setup_cuda_device(eval_config.gpu)
    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
        torch.set_float32_matmul_precision("high")
        logger.info("Using tfloat32 high precision matmul")

    if eval_config.custom_run_suffix:
        run_name += f"__{eval_config.custom_run_suffix}"

    if eval_config.log_wandb:
        project = specific_model_to_wandb_project(specific_model_config).value
        wandb.init(project=project, name=f"eval_{run_name}")
        wandb.config.update(eval_config)  # type: ignore
        wandb.config.update(outer_code_config)  # type: ignore
        wandb.config.update(channel_config)  # type: ignore
        wandb.config.update(inner_code_wandb_config)  # type: ignore
        wandb.config.update(
            {
                "specific_model": specific_model_identifier,
                "ssh_server_name": get_ssh_server_name(logger),
            },
            allow_val_change=True,  # Required because Specific model is already set but in an unreadable way
        )

    evaluate_model(
        eval_config,
        specific_model_config,
        channel_config,
        logger,
        code_config.outer_code_config,
        code_config.inner_code_config,
    )

    if eval_config.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
