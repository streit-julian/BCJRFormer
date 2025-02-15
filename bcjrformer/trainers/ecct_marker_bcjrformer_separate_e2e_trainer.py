import logging
import os

import torch
from bcjrformer.codes.linear_code import LinearCode
from bcjrformer.codes.marker_code import MarkerCode
from bcjrformer.configs.channel_config import IDSChannelConfig
from bcjrformer.configs.model_config import ModelConfig
from torch.utils.data import DataLoader
from bcjrformer.configs.specific_model_config import EcctOnBCJRFormerE2EModel
from bcjrformer.datasets.bcjrformer_dataset import (
    BCJRFormerMarkerDataset,
)
from bcjrformer.models.bcjrformer import (
    BCJRFormer,
)
from bcjrformer.models.ids_ecct_outer_model import IdsEcctOuterTransformer
from bcjrformer.trainers.base_trainer import BaseTrainer
from bcjrformer.utils import (
    BER,
    FER,
    bin_to_sign,
    repair_compiled_state_dict,
    setup_logger,
    sign_to_bin,
)


class ECCTMarkerBCJRFormerSeparateE2ETrainer(BaseTrainer[BCJRFormerMarkerDataset]):
    def __init__(
        self,
        model_config: ModelConfig,
        ecct_on_bcjrformer_e2e_config: EcctOnBCJRFormerE2EModel,
        window_block_dimension: int,
        linear_code: LinearCode,
        marker_code: MarkerCode,
        channel_config: IDSChannelConfig,
        device: torch.device,
        logger: logging.Logger | None = None,
    ) -> None:

        self.inner_pre_trained_model_path = (
            ecct_on_bcjrformer_e2e_config.inner_pre_trained_model_path
        )
        self.marker_code = marker_code
        self.linear_code = linear_code

        self.pc_matrix = (
            torch.tensor(self.linear_code.pc_matrix)
            .transpose(0, 1)
            .float()
            .to(self.device)
        )

        inner_model_dict = self.load_inner_model_dict(self.inner_pre_trained_model_path)
        self.logger.info(
            f"Inner Model Loss: {inner_model_dict['loss']}, Epoch: {inner_model_dict['epoch']}"
        )
        self.inner_model: BCJRFormer = (
            BCJRFormer(
                model_config=inner_model_dict["config"],
                window_block_dimension=window_block_dimension,
                encoded_length=marker_code.encoded_length,
                device=device,
            )
            .to(device)
            .requires_grad_(False)  # Freeze the pre-trained inner model
        )
        self.inner_model.load_state_dict(inner_model_dict["model_state_dict"])

        self.inner_model.eval()

        model: IdsEcctOuterTransformer = IdsEcctOuterTransformer(
            model_config=model_config,
            code=linear_code,
            device=device,
        )

        logger = setup_logger(__name__, logger)

        logger.info(f"Model (Inner): {self.inner_model}")

        super().__init__(
            model_config=model_config,
            model=model,
            device=device,
            logger=logger,
        )

    def load_inner_model_dict(self, inner_model_path: str | None) -> dict:
        if inner_model_path is None:
            raise ValueError("Inner model path is not set. Cannot load inner model")

        if not os.path.exists(inner_model_path):
            raise ValueError(
                f"Inner model path {inner_model_path} does not exist. Cannot load inner model"
            )

        model_dict = torch.load(inner_model_path)

        self.logger.info(f"Loaded inner model from {inner_model_path}")

        if model_dict["config"].compile_model:
            self.logger.info("Repairing compiled state dict")
            repaired_model_state_dict = repair_compiled_state_dict(
                model_dict["model_state_dict"]
            )
            model_dict["model_state_dict"] = repaired_model_state_dict

        return model_dict

    def inner_decode(self, x, y, padding_mask):
        x_inner_pred = self.inner_model(y.to(self.device), padding_mask.to(self.device))

        x_from_inner_ll = torch.sigmoid(x_inner_pred)

        x_to_outer_ll, _ = self.marker_code.decode_model(x_from_inner_ll)

        x_to_outer_ll_sign = bin_to_sign(x_to_outer_ll)

        z_mul = x_to_outer_ll_sign * bin_to_sign(x.to(self.device))

        y_magnitude = torch.abs(x_to_outer_ll_sign)

        y_syndrome = (
            torch.matmul(
                sign_to_bin(torch.sign(x_to_outer_ll_sign)),
                self.pc_matrix,
            ).long()
            % 2
        )

        y_syndrome = bin_to_sign(y_syndrome)
        return x_to_outer_ll_sign, z_mul, y_magnitude, y_syndrome

    def epoch_train(
        self,
        train_loader: DataLoader[BCJRFormerMarkerDataset],
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, dict]:
        self.model.train()

        cum_loss = cum_ber_inner = cum_fer_inner = cum_samples = cum_ber = cum_fer = 0.0

        for batch_idx, (
            m,
            x,
            _,
            _,
            y,
            padding_mask,
        ) in enumerate(train_loader):

            x_to_outer_ll_sign, z_mul, y_magnitude, y_syndrome = self.inner_decode(
                x, y, padding_mask
            )

            z_pred = self.model(y_magnitude, y_syndrome)

            loss, x_pred = self.model.loss(-z_pred, z_mul, x_to_outer_ll_sign)

            loss.backward()

            if ((batch_idx + 1) % self.model_config.batch_accumulation == 0) or (
                batch_idx == (len(train_loader) - 1)
            ):
                optimizer.step()
                self.model.zero_grad()

            with torch.no_grad():
                # x_pred = torch.round(torch.sigmoid(x_pred))
                x_from_inner_pred = torch.round(sign_to_bin(x_to_outer_ll_sign))

                x_dev = x.to(self.device)

                ber = BER(x_pred, x_dev)
                fer = FER(x_pred, x_dev)
                ber_inner = BER(x_from_inner_pred, x_dev)
                fer_inner = FER(x_from_inner_pred, x_dev)

                cum_loss += loss.item() * x.shape[0]
                cum_ber += ber * x.shape[0]
                cum_fer += fer * x.shape[0]
                cum_ber_inner += ber_inner * x.shape[0]
                cum_fer_inner += fer_inner * x.shape[0]
                cum_samples += x.shape[0]
                if (batch_idx + 1) % 500 == 0 or batch_idx == (len(train_loader) - 1):
                    self.logger.info(
                        f"Train epoch {self.epoch} "
                        + f"Batch {batch_idx + 1}/{len(train_loader)}: "
                        + f"Loss={cum_loss / cum_samples:.2e} "
                        + f"BER={cum_ber / cum_samples:.2e} "
                        + f"FER={cum_fer / cum_samples:.2e}"
                        + f"BER_INNER={cum_ber_inner / cum_samples:.2e} "
                        + f"FER_INNER={cum_fer_inner / cum_samples:.2e} "
                    )

        log_dict = {
            "Train Loss (Epoch)": cum_loss / cum_samples,
            "Train BER (Epoch)": cum_ber / cum_samples,
            "Train FER (Epoch)": cum_fer / cum_samples,
            "Train Inner BER (Epoch)": cum_ber_inner / cum_samples,
            "Train Inner FER (Epoch)": cum_fer_inner / cum_samples,
        }

        return cum_loss / cum_samples, log_dict

    def epoch_evaluate(self, test_loader: DataLoader) -> tuple[float, dict]:
        cum_count = test_ber = test_fer = test_ber_inner = test_loss = (
            test_fer_inner
        ) = 0.0
        for (
            _,
            x,
            x_i,
            x_pred_bcjr,
            y,
            padding_mask,
        ) in iter(test_loader):
            x_from_inner_pred, z_mul, y_magnitude, y_syndrome = self.inner_decode(
                x, y, padding_mask
            )

            z_pred = self.model(y_magnitude, y_syndrome)

            loss, x_pred = self.model.loss(-z_pred, z_mul, x_from_inner_pred)

            test_loss += loss.item() * x.shape[0]

            x_dev = x.to(self.device)

            test_ber += BER(x_pred, x_dev) * x.shape[0]
            test_fer += FER(x_pred, x_dev) * x.shape[0]

            test_ber_inner += BER(x_from_inner_pred, x_dev) * x.shape[0]
            test_fer_inner += FER(x_from_inner_pred, x_dev) * x.shape[0]

            cum_count += x.shape[0]

        test_loss /= cum_count
        test_ber /= cum_count
        test_fer /= cum_count
        test_ber_inner /= cum_count
        test_fer_inner /= cum_count

        self.logger.info(
            "Test Loss: {:.2e}, Test BER: {:.2e}, Test FER: {:.2e}".format(
                test_loss, test_ber, test_fer
            )
        )
        log_dict = {
            "Test Loss (Epoch)": test_loss,
            "Test BER (Epoch)": test_ber,
            "Test FER (Epoch)": test_fer,
            "Test BER Inner (Epoch)": test_ber_inner,
            "Test FER Inner (Epoch)": test_fer_inner,
        }

        return test_loss, log_dict
