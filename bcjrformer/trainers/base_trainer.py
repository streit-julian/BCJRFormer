from __future__ import annotations
import os
import time
import logging
from abc import abstractmethod
from typing import Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import wandb

from bcjrformer.configs.model_config import ModelConfig
from typing import TypeVar

from bcjrformer.utils import setup_logger

T = TypeVar("T", bound=Dataset)


class BaseTrainer[T]:
    def __init__(
        self,
        model_config: ModelConfig,
        model: torch.nn.Module,
        device: torch.device,
        logger: logging.Logger | None = None,
        log_interval: int = 500,
    ) -> None:
        self.logger = logger if logger is not None else setup_logger(__name__)
        self.model_config = model_config
        self.model = model.to(device)

        if model_config.compile_model:
            self.logger.info("Compiling model...")
            self.model: torch.nn.Module = torch.compile(self.model)  # type: ignore
            self.logger.info("Model compiled.")

        self.device: torch.device = device

        self.batch_accumulation: int = model_config.batch_accumulation

        # Checkpoint Options
        self.save_checkpoints_every_n_epochs = (
            model_config.save_checkpoints_every_n_epochs
        )
        self.from_checkpoint = model_config.from_checkpoint

        self.log_wandb = model_config.log_wandb

        self.log_interval: int = log_interval

        self.epochs = model_config.epochs

        self.logger.info(f"Model: {self.model}")

        n_params = np.sum([np.prod(p.shape) for p in self.model.parameters()])
        trainable_params = np.sum(
            [np.prod(p.shape) for p in self.model.parameters() if p.requires_grad]
        )
        self.logger.info(
            f"Number of parameters:{n_params} (trainable: {trainable_params})"
        )

    def evaluate(self, dataloader: DataLoader[Any]) -> tuple[float, dict]:

        if len(dataloader) == 0:
            raise ValueError("Evaluation data loader is empty")

        self.model.eval()
        with torch.no_grad():
            loss, log_dict = self.epoch_evaluate(dataloader)

            self.logger.info(f"Evaluation Config: {log_dict}")

            if self.log_wandb:
                wandb.log(log_dict)

        return loss, log_dict

    def train(
        self,
        model_dir: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
    ) -> None:

        if len(train_loader) == 0:
            raise ValueError("Training data loader is empty")

        best_loss: float = float("inf")

        start_epoch = 1

        if self.from_checkpoint:
            checkpoint = self.get_checkpoint()
            best_loss = (
                checkpoint["best_loss"]
                if "best_loss" in checkpoint
                else checkpoint["loss"]
            )
            start_epoch = checkpoint["epoch"]

            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.logger.info(f"Starting training at epoch: {start_epoch}")

        self.epoch = start_epoch

        self.model.train()

        for self.epoch in range(start_epoch, self.epochs + 1):

            self.logger.info(f"Training Epoch {self.epoch}")

            epoch_loss: float
            epoch_log_dict: dict

            epoch_start = time.time()
            epoch_loss, epoch_log_dict = self.epoch_train(train_loader, optimizer)

            epoch_log_dict["Epoch"] = self.epoch

            self.logger.info(
                f"Epoch {self.epoch} - Time: {time.time() - epoch_start}, Loss: {epoch_loss:.2e} "
            )

            is_best_model = epoch_loss < best_loss

            if is_best_model:
                best_loss = epoch_loss
                self.save_checkpoint(
                    self.epoch,
                    epoch_loss,
                    optimizer,
                    scheduler,
                    best_loss,
                    f"{model_dir}/best_model.pth",
                )

            if (
                self.save_checkpoints_every_n_epochs is not None
                and self.epoch % self.save_checkpoints_every_n_epochs == 0
            ):
                self.save_checkpoint(
                    self.epoch,
                    epoch_loss,
                    optimizer,
                    scheduler,
                    best_loss,
                    f"{model_dir}/model_checkpoint_{self.epoch}.pth",
                )

            if self.log_wandb:
                wandb.log(epoch_log_dict)

            self.logger.info(epoch_log_dict)

            scheduler.step()

    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        best_loss,
        path: str,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss,
                "best_loss": best_loss,
            },
            path,
        )

    def get_checkpoint(self) -> Any:
        from_checkpoint_epoch = self.model_config.from_checkpoint_epoch
        checkpoint_dir = self.model_config.checkpoint_dir

        # generally we already validate this in the config but for safety we
        # double check
        if checkpoint_dir is None:
            raise ValueError("Checkpoint directory is not set")

        if from_checkpoint_epoch is not None:
            checkpoint_path = (
                f"{checkpoint_dir}/model_checkpoint_{from_checkpoint_epoch}.pth"
            )
        else:
            checkpoint_path = f"{checkpoint_dir}/best_model.pth"

        if not os.path.exists(checkpoint_path):
            raise ValueError(
                "Checkpoint file doesn't exist. Cannot train from checkpoint {}".format(
                    checkpoint_path
                )
            )

        return torch.load(checkpoint_path)

    @abstractmethod
    def epoch_train(
        self,
        train_loader: DataLoader[T],
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, dict]:
        """Train one training epoch. Must return a tuple (loss, log_dict)."""
        raise NotImplementedError("Subclasses must implement train_step.")

    @abstractmethod
    def epoch_evaluate(self, test_loader: DataLoader[T]) -> tuple[float, dict]:
        """Evaluate one training epoch. Must return  a tuple (loss, log dict)."""
        raise NotImplementedError("Subclasses must implement log_step.")
