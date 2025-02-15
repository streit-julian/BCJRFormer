from importlib.resources import files
import logging
import os
import random
import numpy as np
import bcjrformer
import bcjrformer.code_db
import pathlib
from typing import cast
import torch
from bcjrformer.codes.galois_field import GaloisField2m

from bcjrformer.configs.evaluation_config import EvaluationConfig
from bcjrformer.configs.model_config import ModelConfig


def setup_cuda_device(gpu: int):
    """

    Args:
        gpu (int): _description_
    """
    if gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def get_package_dir() -> pathlib.PosixPath:
    return cast(pathlib.PosixPath, files(bcjrformer))


def get_code_dir() -> pathlib.PosixPath:
    return cast(pathlib.PosixPath, files(bcjrformer.code_db))


def bin_to_sign(x):
    return 1 - 2 * x


def sign_to_bin(x):
    return 0.5 * (1 - x)


def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()


def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calculate_standard_form_bin(
    matrix: np.ndarray, cols: int | None = None
) -> np.ndarray:
    gf = GaloisField2m(2)

    matrix = matrix.copy()

    cols = cols or matrix.shape[1]

    for j in range(cols):
        if matrix[j, j] == 0:
            for i in range(j + 1, matrix.shape[0]):
                if matrix[i, j] == 1:
                    matrix[[i, j]] = matrix[[j, i]]
                    break

            if matrix[j, j] == 0:
                for h in range(j + 1, matrix.shape[1]):
                    if matrix[j, h] == 1:
                        matrix[:, [j, h]] = matrix[:, [h, j]]
                        break

            if matrix[j, j] == 0:
                raise ValueError("Matrix does not have full rank")

        for i in range(matrix.shape[0]):
            if i != j and matrix[i, j] == 1:
                matrix[i] = gf.add(matrix[i], matrix[j])

    return matrix


def code_exclude_fn(path: str, root: str) -> bool:
    rel_path = os.path.relpath(path, root)
    if rel_path.startswith(".") and rel_path.endswith(os.sep):
        return True

    if rel_path.startswith("wandb" + os.sep):
        return True

    return False


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def repair_compiled_state_dict(in_state_dict: dict):
    pairings = [
        (src_key, remove_prefix(src_key, "_orig_mod."))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return in_state_dict
    out_state_dict = {}
    for src_key, dest_key in pairings:
        out_state_dict[dest_key] = in_state_dict[src_key]
    return out_state_dict


def setup_logger(
    name,
    parent: logging.Logger | None = None,
    extra_handlers: list[logging.Handler] | None = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name) if parent is None else parent.getChild(name)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if extra_handlers:
        for handler in extra_handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    return logger


def get_ssh_server_name(logger: logging.Logger):
    ssh_server_name = os.getenv("SSH_SERVER_NAME")

    if ssh_server_name is None:
        logger.warning("No SSH_SERVER_NAME environment variable found.")
        ssh_server_name = "unknown"


def evaluation_config_to_model_config(eval_cfg: EvaluationConfig) -> ModelConfig:
    return ModelConfig(
        specific_model=eval_cfg.specific_model,
        ids_channel=eval_cfg.ids_channel,
        inner_model_config=eval_cfg.inner_model_config,
        masked_attention=eval_cfg.masked_attention,
        workers=eval_cfg.workers,
        gpu=eval_cfg.gpu,
        test_batch_size=eval_cfg.test_batch_size,
        test_batches_per_epoch=eval_cfg.test_batches_per_epoch,
        compile_model=eval_cfg.compile_model,
        N_dec=eval_cfg.N_dec,
        d_model=eval_cfg.d_model,
        h=eval_cfg.h,
        seed=eval_cfg.seed if eval_cfg.seed is not None else 42,
        log_wandb=eval_cfg.log_wandb,
        custom_run_suffix=eval_cfg.custom_run_suffix,
    )
