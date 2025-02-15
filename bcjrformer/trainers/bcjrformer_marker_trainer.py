import logging

import torch
from bcjrformer.codes.marker_code import MarkerCode
from bcjrformer.configs.model_config import ModelConfig
from torch.utils.data import DataLoader
from bcjrformer.datasets.bcjrformer_dataset import (
    BCJRFormerMarkerDataset,
)
from bcjrformer.models.bcjrformer import (
    BCJRFormer,
)
from bcjrformer.trainers.base_trainer import BaseTrainer
from bcjrformer.utils import BER, FER, setup_logger
import torch.nn.functional as F


class BCJRFormerMarkerTrainer(BaseTrainer[BCJRFormerMarkerDataset]):
    example_table = None

    def __init__(
        self,
        model_config: ModelConfig,
        window_block_dimension: int,
        marker_code: MarkerCode,
        device: torch.device,
        logger: logging.Logger | None = None,
    ) -> None:

        self.marker_code = marker_code

        model: BCJRFormer = BCJRFormer(
            model_config=model_config,
            window_block_dimension=window_block_dimension,
            encoded_length=marker_code.encoded_length,
            device=device,
            mask=None,
        )

        logger = setup_logger(__name__, logger)

        super().__init__(
            model_config=model_config,
            model=model,
            device=device,
            logger=logger,
        )

        self.marker_sequence = torch.tensor(self.marker_code.marker, device=self.device)

    def epoch_train(
        self,
        train_loader: DataLoader[BCJRFormerMarkerDataset],
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, dict]:

        if train_loader.batch_size is None:
            raise ValueError(
                "Batch size is not set on the dataloader. This is unexpected."
            )

        x_marker = self.marker_sequence.repeat(
            self.marker_code.encoded_length // self.marker_code.block_length
        ).expand(train_loader.batch_size, -1)
        cum_loss = cum_ber = cum_fer = cum_samples = cum_ber_bcjr = cum_fer_bcjr = (
            cum_marker_ber
        ) = cum_marker_fer = 0.0

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

                x_pred_o, x_marker_pred = self.marker_code.decode_model(x_pred)

                ###
                ber = BER(x_pred_o, x.to(self.device))
                fer = FER(x_pred_o, x.to(self.device))

                marker_ber = BER(x_marker, x_marker_pred)
                marker_fer = FER(x_marker, x_marker_pred)

                ber_bcjr = BER(x_pred_bcjr, x)
                fer_bcjr = FER(x_pred_bcjr, x)

                cum_loss += loss.item() * x.shape[0]
                cum_ber += ber * x.shape[0]
                cum_fer += fer * x.shape[0]
                cum_marker_ber += marker_ber * x.shape[0]
                cum_marker_fer += marker_fer * x.shape[0]
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
                        + f"BER_MARKER={cum_marker_ber / cum_samples:.2e} "
                        + f"FER_MARKER={cum_marker_fer / cum_samples:.2e} "
                        + f"BER_BCJR={cum_ber_bcjr / cum_samples:.2e} "
                        + f"FER_BCJR={cum_fer_bcjr / cum_samples:.2e}"
                    )

        log_dict = {
            "Train Loss (Epoch)": cum_loss / cum_samples,
            "Train BER (Epoch)": cum_ber / cum_samples,
            "Train FER (Epoch)": cum_fer / cum_samples,
            "Train BER Marker (Epoch)": cum_marker_ber / cum_samples,
            "Train FER Marker (Epoch)": cum_marker_fer / cum_samples,
            "Train BER BCJR (Epoch)": cum_ber_bcjr / cum_samples,
            "Train FER BCJR (Epoch)": cum_fer_bcjr / cum_samples,
        }

        return cum_loss / cum_samples, log_dict

    def epoch_evaluate(self, test_loader: DataLoader) -> tuple[float, dict]:

        n_batch = 0

        test_loss = test_ber = test_fer = cum_count = test_ber_bcjr = test_fer_bcjr = (
            test_ber_marker
        ) = test_fer_marker = 0.0
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
            x_pred_o, x_marker_pred = self.marker_code.decode_model(x_pred)

            x_marker = (
                torch.tensor(self.marker_code.marker, device=self.device)
                .repeat(x_marker_pred.shape[1] // self.marker_code.marker_length)
                .expand(x_marker_pred.shape[0], -1)
            )

            test_loss += loss.item() * x.shape[0]

            test_ber += BER(x_pred_o, x.to(self.device)) * x.shape[0]
            test_fer += FER(x_pred_o, x.to(self.device)) * x.shape[0]

            test_ber_marker += BER(x_marker, x_marker_pred) * x.shape[0]
            test_fer_marker += FER(x_marker, x_marker_pred) * x.shape[0]

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
                    + f"BER_MARKER={test_ber_marker / cum_count:.2e} "
                    + f"FER_MARKER={test_fer_marker / cum_count:.2e} "
                    + f"BER BCJR={test_ber_bcjr / cum_count:.2e} "
                    + f"FER BCJR={test_fer_bcjr / cum_count:.2e}"
                )
        test_loss /= cum_count
        test_ber /= cum_count
        test_fer /= cum_count
        test_ber_bcjr /= cum_count
        test_fer_bcjr /= cum_count
        test_ber_marker /= cum_count
        test_fer_marker /= cum_count

        self.logger.info(
            "Test Loss: {:.2e}, Test BER: {:.2e}, Test FER: {:.2e}".format(
                test_loss, test_ber, test_fer
            )
        )
        log_dict = {
            "Test Loss (Epoch)": test_loss,
            "Test BER (Epoch)": test_ber,
            "Test FER (Epoch)": test_fer,
            "Test BER Marker (Epoch)": test_ber_marker,
            "Test FER Marker (Epoch)": test_fer_marker,
            "Test BER BCJR (Epoch)": test_ber_bcjr,
            "Test FER BCJR (Epoch)": test_fer_bcjr,
        }
        return test_loss, log_dict

    def predict(self):
        pass
