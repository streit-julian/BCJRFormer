import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bcjrformer.codes.convolutional_code import LinearConvolutionalCode
from bcjrformer.configs.model_config import ModelConfig
from bcjrformer.configs.specific_model_config import ConvBCJRFormerModel
from bcjrformer.datasets.bcjrformer_dataset import BCJRFormerConvCombinedDataset
from bcjrformer.models.convbcjrformer import ConvBCJRFormer
from bcjrformer.trainers.base_trainer import BaseTrainer
from bcjrformer.utils import BER, FER, setup_logger


def build_mask(conv_code: LinearConvolutionalCode) -> torch.Tensor:
    g = conv_code.generator_matrix

    mask = ~(torch.from_numpy(g.copy()).bool())

    # import matplotlib.pyplot as plt
    # plt.matshow(~(mask).numpy())
    # plt.savefig("cross_mask.png")

    return mask


class ConvBCJRFormerTrainer(BaseTrainer[BCJRFormerConvCombinedDataset]):
    def __init__(
        self,
        model_config: ModelConfig,
        conv_bcjrformer_config: ConvBCJRFormerModel,
        conv_code: LinearConvolutionalCode,
        symbol_window_block_dimension: int,
        state_window_block_dimension: int,
        symbol_n: int,
        state_n: int,
        device: torch.device,
        logger: logging.Logger | None = None,
    ) -> None:
        mask = build_mask(conv_code) if model_config.masked_attention else None

        model: ConvBCJRFormer = ConvBCJRFormer(
            model_config=model_config,
            conv_bcjrformer_config=conv_bcjrformer_config,
            device=device,
            symbol_window_block_dimension=symbol_window_block_dimension,
            state_window_block_dimension=state_window_block_dimension,
            symbol_n=symbol_n,
            state_n=state_n,
            dropout=0,
            generator_mask=mask,
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
        train_loader: DataLoader[BCJRFormerConvCombinedDataset],
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, dict]:
        cum_loss = cum_ber = cum_fer = cum_samples = cum_ber_bcjr = cum_fer_bcjr = (
            cum_ber_inner
        ) = cum_fer_inner = cum_ber_outer = cum_fer_outer = cum_ber_zero_end = (
            cum_fer_zero_end
        ) = 0.0

        for batch_idx, (
            m,
            x,
            x_i,
            x_target,
            x_pred_bcjr,
            y_bit,
            y_state,
            _,
        ) in enumerate(train_loader):

            n = x.shape[1]
            enc_length = x_i.shape[1]

            x_full = torch.cat((x_i, x_target), -1).to(self.device)

            x_pred = self.model(
                y_bit.to(self.device),
                y_state.to(self.device),
            )

            loss = F.binary_cross_entropy_with_logits(x_pred, x_full)

            # self.model.zero_grad()
            loss.backward()

            if ((batch_idx + 1) % self.model_config.batch_accumulation == 0) or (
                batch_idx == (len(train_loader) - 1)
            ):
                optimizer.step()
                self.model.zero_grad()

            with torch.no_grad():
                x_pred = torch.round(torch.sigmoid(x_pred))

                x_pred_inner = x_pred[:, :enc_length]
                x_pred_outer = x_pred[:, enc_length : enc_length + n]
                x_pred_zero_end = x_pred[:, enc_length + n :]

                ###
                ber = BER(x_pred, x_full)
                ber_inner = BER(x_pred_inner, x_i.to(self.device))
                ber_outer = BER(x_pred_outer, x.to(self.device))
                ber_zero_end = BER(
                    x_pred_zero_end, torch.zeros_like(x_pred_zero_end).to(self.device)
                )

                fer = FER(x_pred, x_full)
                fer_inner = FER(x_pred_inner, x_i.to(self.device))
                fer_outer = FER(x_pred_outer, x.to(self.device))
                fer_zero_end = FER(
                    x_pred_zero_end, torch.zeros_like(x_pred_zero_end).to(self.device)
                )

                ber_bcjr = BER(x_pred_bcjr, x)
                fer_bcjr = FER(x_pred_bcjr, x)

                cum_loss += loss.item() * x.shape[0]
                cum_ber += ber * x.shape[0]
                cum_fer += fer * x.shape[0]
                cum_ber_inner += ber_inner * x.shape[0]
                cum_fer_inner += fer_inner * x.shape[0]
                cum_ber_outer += ber_outer * x.shape[0]
                cum_fer_outer += fer_outer * x.shape[0]
                cum_ber_zero_end += ber_zero_end * x.shape[0]
                cum_fer_zero_end += fer_zero_end * x.shape[0]
                cum_ber_bcjr += ber_bcjr * x.shape[0]
                cum_fer_bcjr += fer_bcjr * x.shape[0]
                cum_samples += x.shape[0]
                if (batch_idx + 1) % self.log_interval == 0 or batch_idx == (
                    len(train_loader) - 1
                ):
                    self.logger.info(
                        f"Train epoch {self.epoch} "
                        + f"Batch {batch_idx + 1}/{len(train_loader)}: "
                        + f"Loss={cum_loss / cum_samples:.2e} "
                        + f"BER={cum_ber / cum_samples:.2e} "
                        + f"FER={cum_fer / cum_samples:.2e} "
                        + f"BER_INNER={cum_ber_inner / cum_samples:.2e} "
                        + f"FER_INNER={cum_fer_inner / cum_samples:.2e} "
                        + f"BER_OUTER={cum_ber_outer / cum_samples:.2e} "
                        + f"FER_OUTER={cum_fer_outer / cum_samples:.2e} "
                        + f"BER_ZERO_END={cum_ber_zero_end / cum_samples:.2e} "
                        + f"FER_ZERO_END={cum_fer_zero_end / cum_samples:.2e} "
                        + f"BER_BCJR={cum_ber_bcjr / cum_samples:.2e} "
                        + f"FER_BCJR={cum_fer_bcjr / cum_samples:.2e}"
                    )

        log_dict = {
            "Batch Train Loss (Epoch)": cum_loss / cum_samples,
            "Batch Train BER (Epoch)": cum_ber / cum_samples,
            "Batch Train FER (Epoch)": cum_fer / cum_samples,
            "Batch Train BER Inner (Epoch)": cum_ber_inner / cum_samples,
            "Batch Train FER Inner (Epoch)": cum_fer_inner / cum_samples,
            "Batch Train BER Outer (Epoch)": cum_ber_outer / cum_samples,
            "Batch Train FER Outer (Epoch)": cum_fer_outer / cum_samples,
            "Batch Train BER Zero End (Epoch)": cum_ber_zero_end / cum_samples,
            "Batch Train FER Zero End (Epoch)": cum_fer_zero_end / cum_samples,
            "Batch Train BER BCJR (Epoch)": cum_ber_bcjr / cum_samples,
            "Batch Train FER BCJR (Epoch)": cum_fer_bcjr / cum_samples,
        }

        return cum_loss / cum_samples, log_dict

    def epoch_evaluate(self, test_loader: DataLoader) -> tuple[float, dict]:
        n_batch = 0
        test_loss = test_ber = test_fer = cum_count = test_ber_bcjr = test_fer_bcjr = (
            test_ber_inner
        ) = test_fer_inner = test_ber_outer = test_fer_outer = test_ber_zero_end = (
            test_fer_zero_end
        ) = 0.0
        for (
            m,
            x,
            x_i,
            x_target,
            x_pred_bcjr,
            y_bit,
            y_state,
            _,
        ) in iter(test_loader):
            n = x.shape[1]
            enc_length = x_i.shape[1]
            x_full = torch.cat((x_i, x_target), -1).to(self.device)

            x_pred = self.model(
                y_bit.to(self.device),
                y_state.to(self.device),
            )

            loss = F.binary_cross_entropy_with_logits(x_pred, x_full)

            x_pred = torch.round(torch.sigmoid(x_pred))

            x_pred_inner = x_pred[:, :enc_length]
            x_pred_outer = x_pred[:, enc_length : enc_length + n]
            x_pred_zero_end = x_pred[:, enc_length + n :]

            test_loss += loss.item() * x.shape[0]
            test_ber += BER(x_pred, x_full) * x.shape[0]
            test_fer += FER(x_pred, x_full) * x.shape[0]

            test_ber_inner += BER(x_pred_inner, x_i.to(self.device)) * x.shape[0]
            test_ber_outer += BER(x_pred_outer, x.to(self.device)) * x.shape[0]
            test_ber_zero_end += (
                BER(
                    x_pred_zero_end,
                    torch.zeros_like(x_pred_zero_end).to(self.device),
                )
                * x.shape[0]
            )
            test_fer_inner += FER(x_pred_inner, x_i.to(self.device)) * x.shape[0]
            test_fer_outer += FER(x_pred_outer, x.to(self.device)) * x.shape[0]
            test_fer_zero_end += (
                FER(
                    x_pred_zero_end,
                    torch.zeros_like(x_pred_zero_end).to(self.device),
                )
                * x.shape[0]
            )

            test_ber_bcjr += BER(x_pred_bcjr, x) * x.shape[0]
            test_fer_bcjr += FER(x_pred_bcjr, x) * x.shape[0]

            cum_count += x.shape[0]
            n_batch += 1

            if n_batch % 500 == 0:
                self.logger.info(
                    f"Test Batch {n_batch}/{len(test_loader)}: "
                    + f"Loss={test_loss / cum_count:.2e} "
                    + f"BER={test_ber / cum_count:.2e} "
                    + f"FER={test_fer / cum_count:.2e} "
                    + f"BER_INNER={test_ber_inner / cum_count:.2e} "
                    + f"FER_INNER={test_fer_inner / cum_count:.2e} "
                    + f"BER_OUTER={test_ber_outer / cum_count:.2e} "
                    + f"FER_OUTER={test_fer_outer / cum_count:.2e} "
                    + f"BER_BCJR={test_ber_bcjr / cum_count:.2e} "
                    + f"FER_BCJR={test_fer_bcjr / cum_count:.2e}"
                )

        test_loss /= cum_count
        test_ber /= cum_count
        test_fer /= cum_count
        test_ber_inner /= cum_count
        test_fer_inner /= cum_count
        test_ber_outer /= cum_count
        test_fer_outer /= cum_count
        test_ber_zero_end /= cum_count
        test_fer_zero_end /= cum_count
        test_ber_bcjr /= cum_count
        test_fer_bcjr /= cum_count

        log_dict = {
            "Test Loss (Epoch)": test_loss,
            "Test BER (Epoch)": test_ber,
            "Test FER (Epoch)": test_fer,
            "Test BER Inner (Epoch)": test_ber_inner,
            "Test FER Inner (Epoch)": test_fer_inner,
            "Test BER Outer (Epoch)": test_ber_outer,
            "Test FER Outer (Epoch)": test_fer_outer,
            "Test BER Zero End (Epoch)": test_ber_zero_end,
            "Test FER Zero End (Epoch)": test_fer_zero_end,
            "Test BER BCJR (Epoch)": test_ber_bcjr,
            "Test FER BCJR (Epoch)": test_fer_bcjr,
        }

        return test_loss, log_dict
