import logging
import time

import torch
import wandb
from bcjrformer.configs.channel_config import IDSChannelConfig
from bcjrformer.configs.model_config import ModelConfig
from torch.utils.data import DataLoader
from bcjrformer.datasets.bcjrformer_dataset import (
    BCJRFormerConvDataset,
)
from bcjrformer.models.bcjrformer import (
    BCJRFormer,
)
from bcjrformer.trainers.base_trainer import BaseTrainer
from bcjrformer.utils import BER, FER, setup_logger
import torch.nn.functional as F


class BCJRFormerTrainer(BaseTrainer[BCJRFormerConvDataset]):
    example_table = None

    def __init__(
        self,
        model_config: ModelConfig,
        window_block_dimension: int,
        encoded_length: int,
        channel_config: IDSChannelConfig,
        device: torch.device,
        logger: logging.Logger | None = None,
    ) -> None:

        self.ids_channel_config = channel_config
        self.device = device

        self.train_example_table: wandb.Table | None = None
        self.test_example_table: wandb.Table | None = None

        # checkpoint options
        self.save_checkpoints_every_n_epochs = (
            model_config.save_checkpoints_every_n_epochs
        )
        self.from_checkpoint = model_config.from_checkpoint

        self.log_wandb = model_config.log_wandb
        # TODO: Config this

        self.epoch = -1

        model: BCJRFormer = BCJRFormer(
            model_config=model_config,
            device=device,
            window_block_dimension=window_block_dimension,
            encoded_length=encoded_length,
        ).to(device)

        logger = setup_logger(__name__, logger)

        super().__init__(
            model_config=model_config,
            model=model,
            device=device,
            logger=logger,
        )
        self.logger.info(
            f"Window block size: {window_block_dimension}; "
            + f"Std-Mult: {model_config.inner_model_config.delta_std_multiplier}"
        )

    def epoch_train(
        self,
        train_loader: DataLoader[BCJRFormerConvDataset],
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, dict]:
        cum_loss = cum_ber = cum_fer = cum_samples = cum_ber_bcjr = cum_fer_bcjr = 0.0

        t = time.time()
        for batch_idx, (
            m,
            x,
            x_i,
            x_pred_bcjr,
            y,
            padding_mask,
        ) in enumerate(train_loader):

            x_pred = self.model(y.to(self.device), padding_mask.to(self.device))

            loss = F.binary_cross_entropy_with_logits(x_pred, x_i.to(self.device))

            # self.model.zero_grad()
            loss.backward()

            if ((batch_idx + 1) % self.model_config.batch_accumulation == 0) or (
                batch_idx == (len(train_loader) - 1)
            ):
                optimizer.step()
                self.model.zero_grad()

            with torch.no_grad():
                x_pred = torch.round(torch.sigmoid(x_pred))

                ###
                ber = BER(x_pred, x_i.to(self.device))
                fer = FER(x_pred, x_i.to(self.device))

                ber_bcjr = BER(x_pred_bcjr, x)
                fer_bcjr = FER(x_pred_bcjr, x)

                cum_loss += loss.item() * x.shape[0]
                cum_ber += ber * x.shape[0]
                cum_fer += fer * x.shape[0]
                cum_ber_bcjr += ber_bcjr * x.shape[0]
                cum_fer_bcjr += fer_bcjr * x.shape[0]
                cum_samples += x.shape[0]
                if (batch_idx + 1) % 500 == 0 or batch_idx == (len(train_loader) - 1):
                    self.logger.info(
                        f"Train epoch {self.epoch} "
                        + f"Batch {batch_idx + 1}/{len(train_loader)}: "
                        + f"Loss={cum_loss / cum_samples:.2e} "
                        + f"BER={cum_ber / cum_samples:.2e} "
                        + f"FER={cum_fer / cum_samples:.2e} "
                        + f"BER_BCJR={cum_ber_bcjr / cum_samples:.2e} "
                        + f"FER_BCJR={cum_fer_bcjr / cum_samples:.2e}"
                    )

        self.logger.info(f"Epoch {self.epoch} Train Time {time.time() - t}s\n")

        log_dict = {
            "Train Loss (Epoch)": cum_loss / cum_samples,
            "Train BER (Epoch)": cum_ber / cum_samples,
            "Train FER (Epoch)": cum_fer / cum_samples,
            "Train BER BCJR (Epoch)": cum_ber_bcjr / cum_samples,
            "Train FER BCJR (Epoch)": cum_fer_bcjr / cum_samples,
        }

        return (
            cum_loss / cum_samples,
            log_dict,
        )

    def epoch_evaluate(self, test_loader: DataLoader) -> tuple[float, dict]:
        n_batch = 0
        test_loss = test_ber = test_fer = cum_count = test_ber_bcjr = test_fer_bcjr = (
            0.0
        )
        for (
            m,
            x,
            x_i,
            x_pred_bcjr,
            y,
            padding_mask,
        ) in iter(test_loader):
            x_pred = self.model(y.to(self.device), padding_mask.to(self.device))

            loss = F.binary_cross_entropy_with_logits(x_pred, x_i.to(self.device))

            x_pred = torch.round(torch.sigmoid(x_pred))

            test_loss += loss.item() * x.shape[0]

            test_ber += BER(x_pred, x_i.to(self.device)) * x.shape[0]
            test_fer += FER(x_pred, x_i.to(self.device)) * x.shape[0]

            test_ber_bcjr += BER(x_pred_bcjr, x) * x.shape[0]
            test_fer_bcjr += FER(x_pred_bcjr, x) * x.shape[0]

            cum_count += x.shape[0]
            n_batch += 1

            if n_batch % self.log_interval == 0:
                self.logger.info(
                    f"Test Batch {n_batch}/{len(test_loader)}: "
                    + f"Loss={test_loss / cum_count:.2e} "
                    + f"BER={test_ber / cum_count:.2e} "
                    + f"FER={test_fer / cum_count:.2e} "
                    + f"BER_BCJR={test_ber_bcjr / cum_count:.2e} "
                    + f"FER_BCJR={test_fer_bcjr / cum_count:.2e}"
                )

        test_loss /= cum_count
        test_ber /= cum_count
        test_fer /= cum_count
        test_ber_bcjr /= cum_count
        test_fer_bcjr /= cum_count

        self.logger.info(
            "Test Loss: {:.2e}, Test BER: {:.2e}, Test FER: {:.2e}".format(
                test_loss, test_ber, test_fer
            )
        )
        log_dict = {
            "Test Loss (Epoch)": test_loss,
            "Test BER (Epoch)": test_ber,
            "Test FER (Epoch)": test_fer,
            "Test BER BCJR (Epoch)": test_ber_bcjr,
            "Test FER BCJR (Epoch)": test_fer_bcjr,
        }
        return test_loss, log_dict

    def predict(self):
        pass
