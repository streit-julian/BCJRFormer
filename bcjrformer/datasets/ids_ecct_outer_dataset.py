import numpy as np

from bcjrformer.codes.linear_code import LinearCode
from bcjrformer.codes.marker_code import MarkerCode

# from dna_ecct.codes.decoding_params import (
#     create_decoding_params,
# )
from bcjrformer.codes.galois_field import GaloisField2m
from bcjrformer.utils import (
    bin_to_sign,
    sign_to_bin,
)

from bcjrformer.codes.convolutional_code import (
    LinearConvolutionalCode,
    MarkerConvolutionalCode,
)
from bcjrformer.bcjr.cc_bcjr_ids_ldpc import create_decoding_params
import torch
from torch.utils import data

from bcjrformer.channels import IDSChannel
import logging

from numba.core.errors import NumbaPerformanceWarning

import warnings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IdsConvEcctOuterDataset(data.Dataset):
    """CC Forward-Backward Inner Decoder with ECC Transformer as Outer Decoder Dataset"""

    def __init__(
        self,
        code: LinearCode,
        conv_code: LinearConvolutionalCode,
        p_i: float,
        p_d: float,
        p_s: float,
        batch_size: int,
        batches_per_epoch: int,
        use_zero_cw: bool = False,
    ):

        self.linear_code = code

        self.gf = GaloisField2m(self.linear_code.q)

        self.conv_code = conv_code

        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.generator_matrix = code.generator_matrix
        self.pc_matrix = torch.tensor(code.pc_matrix).T

        # a priori inner code
        L_ai = np.zeros((self.linear_code.n + self.conv_code.m, 2))
        L_ai[self.linear_code.n :, 1] = -np.inf

        L_ai[: self.linear_code.n, 1] = np.log(1 / code.q)
        L_ai[: self.linear_code.n, 0] = np.log(1 / code.q)
        self.L_ai = L_ai

        # if self.fixed_del_count is not None:
        #     channel = FixedDeletionChannel(
        #         q=2,
        #         del_count=self.fixed_del_count,
        #         p_s=p_s,
        #     )
        # else:
        channel = IDSChannel(
            q=code.q,
            p_i=p_i,
            p_d=p_d,
            p_s=p_s,
        )

        self.channel = channel

        self.max_insertions_per_symbol = 2

        self.zero_word = np.zeros((self.linear_code.k)) if use_zero_cw else None
        self.zero_cw = np.zeros((self.linear_code.n)) if use_zero_cw else None

    def get_conv_code_offset(self):
        return self.conv_code.offset

    def set_conv_code_offset(self, offset):
        self.conv_code.offset = offset

    def __len__(self):
        # infinite dataset basically
        return int(self.batches_per_epoch * self.batch_size)

    def __getitem__(self, index):
        if self.zero_cw is None or self.zero_word is None:
            # m = torch.randint(0, 2, (1, self.linear_code.k)).squeeze()
            m = np.random.randint(0, 2, (1, self.linear_code.k)).squeeze()
            x = m @ self.generator_matrix % 2
            # x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_cw

        outer_offset = np.random.randint(0, 2, (self.linear_code.n,))

        gf_permutation = self.gf.add(
            np.tile(np.arange(self.linear_code.q), (self.linear_code.n, 1)),
            outer_offset[:, np.newaxis],
        )

        x_o = self.gf.add(x, outer_offset)

        self.conv_code.new_random_offset()

        enc = self.conv_code.encode(x_o.astype(np.int64))

        r, N = self.channel.numba_transmit(enc, 1)

        dp, r = create_decoding_params(
            self.conv_code, r, N, self.channel, self.max_insertions_per_symbol
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="np.dot() is faster on contiguous arrays",
                category=NumbaPerformanceWarning,
            )

            L_ei = self.conv_code.full_numba_bcjr_ids(r, self.L_ai, dp, self.channel)[
                : self.linear_code.n
            ]

        # normalize and transfer to probability space
        L_ei = np.exp(np.apply_along_axis(self.linear_code.log_normalize, 1, L_ei))

        # remove outer offset
        L_ei = L_ei[np.arange(self.linear_code.n)[:, np.newaxis], gf_permutation]

        x_pred_bcjr = L_ei.argmax(axis=1)[: self.linear_code.n]

        # move to signed domain (i. e. 0 -> 1, 1 -> -1, 0.5 -> 0)
        y = bin_to_sign(torch.Tensor(L_ei[: self.linear_code.n, 0]))

        assert not y.isnan().any(), "y is NaN somehow. Needs to be fixed."

        # create magnitude + syndrome for ECCT Outer Decoder

        magnitude = torch.abs(y)

        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(), self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)

        return (
            torch.tensor(m).float(),
            torch.tensor(x).float(),
            y.float(),
            torch.Tensor(x_pred_bcjr).float(),
            magnitude.float(),
            syndrome.float(),
        )


class IdsMarkerEcctOuterDataset(data.Dataset):
    """CC Forward-Backward Inner Decoder with ECC Transformer as Outer Decoder Dataset"""

    def __init__(
        self,
        code: LinearCode,
        marker_code: MarkerCode,
        conv_code: MarkerConvolutionalCode,
        p_i: float,
        p_d: float,
        p_s: float,
        batch_size: int,
        batches_per_epoch: int,
        use_zero_cw: bool = False,
    ):
        self.linear_code = code

        self.gf = GaloisField2m(self.linear_code.q)

        self.marker_code = marker_code

        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.generator_matrix = code.generator_matrix
        self.pc_matrix = torch.tensor(code.pc_matrix.T)
        self.conv_code = conv_code
        self.L_ai = self.marker_code.log_likelihoods

        channel = IDSChannel(
            q=code.q,
            p_i=p_i,
            p_d=p_d,
            p_s=p_s,
        )

        self.channel = channel

        self.max_insertions_per_symbol = 2

        self.zero_word = np.zeros((self.linear_code.k)) if use_zero_cw else None
        self.zero_cw = np.zeros((self.linear_code.n)) if use_zero_cw else None

    def __len__(self):
        # infinite dataset basically
        return int(self.batches_per_epoch * self.batch_size)

    def __getitem__(self, index):
        if self.zero_cw is None or self.zero_word is None:
            # m = torch.randint(0, 2, (1, self.linear_code.k)).squeeze()
            m = np.random.randint(0, 2, (1, self.linear_code.k)).squeeze()
            x = m @ self.generator_matrix % 2
            # x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_cw
        outer_offset = np.random.randint(0, 2, (self.linear_code.n,))

        gf_permutation = self.gf.add(
            np.tile(np.arange(self.linear_code.q), (self.linear_code.n, 1)),
            outer_offset[:, np.newaxis],
        )

        x_o = self.gf.add(x, outer_offset)

        enc = self.marker_code.numba_encode(x_o[np.newaxis]).squeeze()

        r, N = self.channel.numba_transmit(enc, 1)

        dp, r = create_decoding_params(
            self.conv_code, r, N, self.channel, self.max_insertions_per_symbol
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="np.dot() is faster on contiguous arrays",
                category=NumbaPerformanceWarning,
            )

            L_ei = self.conv_code.full_numba_bcjr_ids(r, self.L_ai, dp, self.channel)

        # normalize and transfer to probability space
        L_ei = np.exp(np.apply_along_axis(self.linear_code.log_normalize, 1, L_ei))

        L_ei = self.marker_code.decode(L_ei)

        # remove outer offset
        L_ei = L_ei[np.arange(self.linear_code.n)[:, np.newaxis], gf_permutation]

        x_pred_bcjr = L_ei.argmax(axis=1)[: self.linear_code.n]

        # move to signed domain (i. e. 0 -> 1, 1 -> -1, 0.5 -> 0)
        y = bin_to_sign(torch.Tensor(L_ei[: self.linear_code.n, 0]))

        assert not y.isnan().any(), "y is NaN somehow. Needs to be fixed."

        # create magnitude + syndrome for ECCT Outer Decoder

        magnitude = torch.abs(y)

        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(), self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)

        return (
            torch.tensor(m).float(),
            torch.tensor(x).float(),
            y.float(),
            torch.Tensor(x_pred_bcjr).float(),
            magnitude.float(),
            syndrome.float(),
        )
