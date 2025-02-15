import torch
import torch.nn as nn
from bcjrformer.configs.channel_config import IDSChannelConfig
from bcjrformer.configs.model_config import ModelConfig
from bcjrformer.configs.specific_model_config import CombinedConvBCJRFormerModel
from bcjrformer.models.modules import (
    TransformerEncoder,
)


class CombConvBCJRFormer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        combined_conv_bcjrformer_config: CombinedConvBCJRFormerModel,
        ids_channel_config: IDSChannelConfig,
        bit_window_block_dimension: int,
        state_window_block_dimension: int,
        bit_n: int,
        state_n: int,
        device,
        dropout=0,
        emb_dropout=0,
        mask: None | torch.Tensor = None,
        q=2,
    ):
        super().__init__()

        self.model_config = model_config
        self.ids_channel_config = ids_channel_config
        self.device = device
        self.bit_window_block_dimension = torch.tensor([bit_window_block_dimension]).to(
            device
        )
        self.state_window_block_dimension = torch.tensor(
            [state_window_block_dimension]
        ).to(device)

        self.n_sequence_max = self.model_config.inner_model_config.n_sequence_max

        self.n = bit_n + state_n

        self.mask = mask.to(device) if mask is not None else None

        self.pos_embed = torch.nn.Parameter(torch.empty((self.n, model_config.d_model)))
        self.seq_embed = nn.Embedding(self.n_sequence_max, model_config.d_model)

        self.sequence_to_embed = (
            torch.arange(0, self.n_sequence_max)
            .reshape(-1, 1)
            .multiply(
                torch.ones(self.n * self.n_sequence_max).reshape(
                    self.n_sequence_max, self.n
                )
            )
            .reshape(-1)
            .long()
            .to(device)
        )

        self.embedding_dropout = nn.Dropout(emb_dropout)

        self.decoder = TransformerEncoder(
            model_config.d_model,
            model_config.N_dec,
            model_config.h,
            model_config.dropout,
            mask=self.mask,
        )

        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(model_config.d_model, 1 if q == 2 else q)]
        )

        self.to_patch_embedding_bit = nn.Sequential(
            nn.LayerNorm(bit_window_block_dimension),
            nn.Linear(bit_window_block_dimension, model_config.d_model),
            nn.LayerNorm(model_config.d_model),
        )

        self.to_patch_embedding_state = nn.Sequential(
            nn.LayerNorm(state_window_block_dimension),
            nn.Linear(state_window_block_dimension, model_config.d_model),
            nn.LayerNorm(model_config.d_model),
        )

        # Not sure this is the right thing here, maybe we could keep the dimensions to enforce learning
        # of the convolutional structure
        self.sequence_length_embed = nn.Linear(
            self.n,
            bit_n,
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, y_bit, y_state, padding_mask):
        bs = y_bit.shape[0]

        # [batch_size, n_seq, n, window_size, window_block] -> [batch_size, n_seq *n, window_block]
        y_bit = y_bit.view(bs, y_bit.shape[1] * y_bit.shape[2], -1)

        # [batch_size, n_seq, n, window_size, window_block] -> [batch_size, n_seq *n, window_block]
        y_state = y_state.view(bs, y_state.shape[1] * y_state.shape[2], -1)

        mask = padding_mask.view(bs, -1).unsqueeze(1).unsqueeze(-1)
        # [batch_size, n_seq, n] -> [batch_size, 1, n_seq * n, 1]

        # [batch_size, n_seq * n, window_size] -> [batch_size, n_seq * n, d_model]
        x_bit = self.to_patch_embedding_bit(y_bit)
        x_state = self.to_patch_embedding_state(y_state)

        x = torch.cat([x_bit, x_state], dim=1)

        # [batch_size, n_seq * n, d_model] -> [batch_size, n_seq, n, d_model]
        x = x.view(x.shape[0], -1, self.n, self.model_config.d_model)

        # Add position embeddings (on a per sequence basis)
        x += self.pos_embed

        # [batch_size, n_seq, n, d_model] -> [batch_size, n_seq * n , d_model]
        x = x.view(x.shape[0], -1, self.model_config.d_model)

        # Add sequence embeddings (to distinguish between subsequences)
        x += self.seq_embed(self.sequence_to_embed)

        x = self.embedding_dropout(x)

        x = self.decoder(x, mask)

        x = self.oned_final_embed(x).squeeze(-1)

        if self.n_sequence_max > 1:
            mean_denominator = (
                (padding_mask.sum(axis=-1) == 0).sum(axis=1).unsqueeze(-1)
            )
            mean_nominator = (
                x.view(x.shape[0], -1, self.n).masked_fill(padding_mask, 0).sum(axis=1)
            )

            x = mean_nominator / mean_denominator

        # x = self.sequence_length_embed(x)

        return x


############################################################
############################################################
