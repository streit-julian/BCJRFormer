import logging

import torch
from bcjrformer.codes.linear_code import LinearCode
from bcjrformer.configs.model_config import ModelConfig
from torch.utils.data import DataLoader
from bcjrformer.datasets.ids_ecct_outer_dataset import (
    IdsConvEcctOuterDataset,
    IdsMarkerEcctOuterDataset,
)
from bcjrformer.models.ids_ecct_outer_model import IdsEcctOuterTransformer
from bcjrformer.trainers.base_trainer import BaseTrainer
from bcjrformer.utils import BER, FER, bin_to_sign, setup_logger

OuterDataset = IdsConvEcctOuterDataset | IdsMarkerEcctOuterDataset


class IdsEcctOuterTrainer(BaseTrainer[OuterDataset]):
    def __init__(
        self,
        model_config: ModelConfig,
        code: LinearCode,
        device: torch.device,
        logger: logging.Logger | None,
    ) -> None:
        model: IdsEcctOuterTransformer = IdsEcctOuterTransformer(
            model_config=model_config,
            code=code,
            device=device,
        )

        logger = setup_logger(__name__, logger)

        super().__init__(
            model_config=model_config,
            model=model,
            device=device,
            logger=logger,
        )

    def epoch_train(
        self,
        train_loader: DataLoader[OuterDataset],
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, dict]:
        cum_loss = cum_ber = cum_fer = cum_samples = cum_ber_bcjr = cum_fer_bcjr = 0.0

        for batch_idx, (m, x, y, x_pred_bcjr, magnitude, syndrome) in enumerate(
            train_loader
        ):

            z_mul = y * bin_to_sign(x)
            z_pred = self.model(magnitude.to(self.device), syndrome.to(self.device))

            loss, x_pred = self.model.loss(
                -z_pred, z_mul.to(self.device), y.to(self.device)
            )

            if loss.isnan():
                raise ValueError("Loss is NaN. Exiting training.")

            loss.backward()

            if ((batch_idx + 1) % self.model_config.batch_accumulation == 0) or (
                batch_idx == (len(train_loader) - 1)
            ):
                optimizer.step()
                self.model.zero_grad()

            # x_pred = torch.round(torch.sigmoid(x_pred))

            with torch.no_grad():
                ###
                x_dev = x.to(self.device)
                x_pred_bcjr_dev = x_pred_bcjr.to(self.device)

                ber = BER(x_pred, x_dev)
                fer = FER(x_pred, x_dev)

                ber_bcjr = BER(x_pred_bcjr_dev, x_dev)
                fer_bcjr = FER(x_pred_bcjr_dev, x_dev)

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
                        + f"BER BCJR={cum_ber_bcjr / cum_samples:.2e} "
                        + f"FER BCJR={cum_fer_bcjr / cum_samples:.2e}"
                    )

        log_dict = {
            "Train Loss (Epoch)": cum_loss / cum_samples,
            "Train BER (Epoch)": cum_ber / cum_samples,
            "Train FER (Epoch)": cum_fer / cum_samples,
            "Train Inner BER (Epoch)": cum_ber_bcjr / cum_samples,
            "Train Inner FER (Epoch)": cum_fer_bcjr / cum_samples,
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
        for m, x, y, x_pred_bcjr, magnitude, syndrome in test_loader:

            z_mul = y * bin_to_sign(x)
            z_pred = self.model(magnitude.to(self.device), syndrome.to(self.device))

            loss, x_pred = self.model.loss(
                -z_pred, z_mul.to(self.device), y.to(self.device)
            )

            test_loss += loss.item() * x.shape[0]

            x_dev = x.to(self.device)
            x_pred_bcjr_dev = x_pred_bcjr.to(self.device)

            test_ber += BER(x_pred, x_dev) * x.shape[0]
            test_fer += FER(x_pred, x_dev) * x.shape[0]
            test_ber_bcjr += BER(x_pred_bcjr_dev, x_dev) * x.shape[0]
            test_fer_bcjr += FER(x_pred_bcjr_dev, x_dev) * x.shape[0]
            cum_count += x.shape[0]
            n_batch += 1

            if n_batch % self.log_interval == 0:
                self.logger.info(
                    f"Test Batch {n_batch}/{len(test_loader)}: "
                    + f"Loss={test_loss / cum_count:.2e} "
                    + f"BER={test_ber / cum_count:.2e} "
                    + f"FER={test_fer / cum_count:.2e} "
                    + f"BER BCJR={test_ber_bcjr / cum_count:.2e} "
                    + f"FER BCJR={test_fer_bcjr / cum_count:.2e}"
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
            "Test Inner BER (Epoch)": test_ber_bcjr,
            "Test Inner FER (Epoch)": test_fer_bcjr,
        }

        return test_loss, log_dict
