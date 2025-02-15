import torch
import torch.nn as nn
from bcjrformer.configs.model_config import ModelConfig
from bcjrformer.configs.specific_model_config import ConvBCJRFormerModel
from bcjrformer.models.modules import (
    CrossAttentionLayer,
    TransformerEncoder,
)


class ConvDualAttentionEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        symbol_depth: int,
        symbol_heads: int,
        state_depth: int,
        state_heads: int,
        symbol_to_state_heads: int,
        state_to_symbol_heads: int,
        mask_symbol_to_state: torch.Tensor | None,
        mask_state_to_symbol: torch.Tensor | None,
        dropout: float = 0.0,
        cross_attention_depth: int = 1,
    ):
        super(ConvDualAttentionEncoder, self).__init__()

        self.symbol_encoder = TransformerEncoder(
            model_dim,
            symbol_depth,
            symbol_heads,
            dropout=dropout,
        )

        self.state_encoder = TransformerEncoder(
            model_dim,
            state_depth,
            state_heads,
            dropout=dropout,
        )

        self.cross_attention_layers = nn.ModuleList([])

        for _ in range(cross_attention_depth):
            self.cross_attention_layers.append(
                nn.ModuleList(
                    [
                        CrossAttentionLayer(
                            symbol_to_state_heads,
                            model_dim,
                            dropout=dropout,
                            mask=mask_symbol_to_state,
                        ),
                        CrossAttentionLayer(
                            state_to_symbol_heads,
                            model_dim,
                            dropout=dropout,
                            mask=mask_state_to_symbol,
                        ),
                    ]
                )
            )

    def forward(self, x_symbol, x_state):
        x_symbol = self.symbol_encoder(x_symbol)
        x_state = self.state_encoder(x_state)

        for ca_symbol_to_state, ca_state_to_symbol in self.cross_attention_layers:  # type: ignore
            x_symbol = ca_symbol_to_state(x_symbol, x_state)

            x_state = ca_state_to_symbol(x_state, x_symbol)

        return x_symbol, x_state


class ConvBCJRFormer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        conv_bcjrformer_config: ConvBCJRFormerModel,
        symbol_window_block_dimension: int,
        state_window_block_dimension: int,
        symbol_n: int,
        state_n: int,
        device,
        dropout=0,
        emb_dropout=0,
        generator_mask: None | torch.Tensor = None,
        q=2,
    ):
        super().__init__()

        self.model_config = model_config
        self.device = device
        self.bit_window_block_dimension = torch.tensor(
            [symbol_window_block_dimension]
        ).to(device)
        self.state_window_block_dimension = torch.tensor(
            [state_window_block_dimension]
        ).to(device)

        self.symbol_n = symbol_n
        self.state_n = state_n

        self.generator_mask = (
            generator_mask.to(device) if generator_mask is not None else None
        )

        self.to_patch_embedding_bit = nn.Sequential(
            nn.LayerNorm(symbol_window_block_dimension),
            nn.Linear(symbol_window_block_dimension, model_config.d_model),
            nn.LayerNorm(model_config.d_model),
        )

        self.to_patch_embedding_state = nn.Sequential(
            nn.LayerNorm(state_window_block_dimension),
            nn.Linear(state_window_block_dimension, model_config.d_model),
            nn.LayerNorm(model_config.d_model),
        )

        self.bit_pos_embed = torch.nn.Parameter(
            torch.empty((self.symbol_n, model_config.d_model))
        )

        self.state_pos_embed = torch.nn.Parameter(
            torch.empty((self.state_n, model_config.d_model))
        )
        self.dropout_symbol = nn.Dropout(emb_dropout)
        self.dropout_state = nn.Dropout(emb_dropout)

        self.decoders = nn.ModuleList([])

        for _ in range(model_config.N_dec):
            self.decoders.append(
                ConvDualAttentionEncoder(
                    model_dim=model_config.d_model,
                    symbol_depth=conv_bcjrformer_config.N_dec_symbol,
                    symbol_heads=conv_bcjrformer_config.h_symbol,
                    state_depth=conv_bcjrformer_config.N_dec_state,
                    state_heads=conv_bcjrformer_config.h_state,
                    symbol_to_state_heads=conv_bcjrformer_config.h_symbol_to_state,
                    state_to_symbol_heads=conv_bcjrformer_config.h_state_to_symbol,
                    mask_symbol_to_state=(
                        self.generator_mask.transpose(0, 1)
                        if self.generator_mask is not None
                        else None
                    ),
                    mask_state_to_symbol=self.generator_mask,
                    dropout=dropout,
                    cross_attention_depth=conv_bcjrformer_config.N_dec_cross,
                ),
            )

        self.layer_norm_state = nn.LayerNorm(model_config.d_model)

        self.oned_final_embed_symbol = torch.nn.Sequential(
            nn.LayerNorm(model_config.d_model),
            nn.Linear(model_config.d_model, 1 if q == 2 else q),
        )

        self.oned_final_embed_state = torch.nn.Sequential(
            nn.LayerNorm(model_config.d_model),
            nn.Linear(model_config.d_model, 1 if q == 2 else q),
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, y_symbol, y_state):
        bs = y_symbol.shape[0]

        # [batch_size, n_seq, n, window_size, window_block] -> [batch_size, n_seq *n, window_block]
        y_symbol = y_symbol.view(bs, y_symbol.shape[1] * y_symbol.shape[2], -1)

        # [batch_size, n_seq, n, window_size, window_block] -> [batch_size, n_seq *n, window_block]
        y_state = y_state.view(bs, y_state.shape[1] * y_state.shape[2], -1)

        # [batch_size, n_seq * n, window_size] -> [batch_size, n_seq * n, d_model]
        x_symbol = self.to_patch_embedding_bit(y_symbol)
        x_state = self.to_patch_embedding_state(y_state)

        # Add position embeddings (on a per sequence basis)
        x_symbol += self.bit_pos_embed
        x_state += self.state_pos_embed

        x_symbol = self.dropout_symbol(x_symbol)
        x_state = self.dropout_state(x_state)

        for decoder in self.decoders:
            x_symbol, x_state = decoder(x_symbol, x_state)

        x_symbol = self.oned_final_embed_symbol(x_symbol)
        x_state = self.oned_final_embed_state(x_state)

        x = torch.cat([x_symbol, x_state], dim=1).squeeze(-1)

        return x
