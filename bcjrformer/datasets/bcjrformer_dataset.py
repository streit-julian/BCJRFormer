import numpy as np
from bcjrformer.codes.convolutional_code import (
    LinearConvolutionalCode,
    MarkerConvolutionalCode,
)
from bcjrformer.codes.linear_code import LinearCode
from bcjrformer.channels import IDSChannel
from bcjrformer.codes.marker_code import MarkerCode

import torch
from torch.utils import data

import galois


import logging

from bcjrformer.codes.decoding_params import create_decoding_params
from bcjrformer.trellis import BCJRFormerTrellis

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BCJRFormerMarkerDataset(data.Dataset):
    def __init__(
        self,
        linear_code: LinearCode,
        marker_code: MarkerCode,
        p_i: float,
        p_d: float,
        p_s: float,
        batch_size: int,
        batches_per_epoch: int,
        compare_bcjr: bool = True,
        std_mult: float = 3.5,
        n_sequence_min=1,
        n_sequence_max=1,
        fixed_n_sequence: int | None = None,
    ):
        self.linear_code = linear_code

        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size

        self.marker_code = marker_code

        self.conv_code = MarkerConvolutionalCode(
            p=marker_code.p,
            T=self.marker_code.encoded_length,
            random_offset=False,
        )
        self.q = self.marker_code.q

        # Use python calculate because otherwise we get issues with their parallel matmul within a parallel dataloader setup
        self.GF = galois.GF(self.q, compile="python-calculate")

        self.generator_matrix = self.GF(linear_code.generator_matrix)
        self.pc_matrix = self.GF(linear_code.pc_matrix)

        channel = IDSChannel(
            q=self.q,
            p_i=p_i,
            p_d=p_d,
            p_s=p_s,
        )
        self.max_insertions_per_symbol = 2
        self.L_ai = self.marker_code.log_likelihoods.copy()

        self.channel = channel

        # TODO: Put in config if necessary
        self.compare_bcjr = compare_bcjr

        self.std_mult = std_mult

        self.n_sequence_min = n_sequence_min
        self.n_sequence_max = n_sequence_max

        self.trellis = BCJRFormerTrellis(
            channel=channel,
            T=self.marker_code.encoded_length,
            n_v=1,
            I_max=self.max_insertions_per_symbol,
            prior_llrs=self.L_ai,
            std_mult=self.std_mult,
            q=self.q,
        )

        self.window_block_dimension = self.trellis.get_block_size()
        self.padding_value = -1

        # Fix the number of sequences while keeping the shape as n_sequence_min/n_sequence_max
        # => Used primarily to compare to single sequence models
        self.fixed_n_sequence = fixed_n_sequence

    def __len__(self):
        # infinite dataset basically
        return int(self.batches_per_epoch * self.batch_size)

    def __getitem__(self, index):

        # return 1

        m = self.GF.Random((1, self.linear_code.k))

        x = m @ self.generator_matrix

        # m = torch.randint(0, self.q, (1, self.linear_code.k))
        # x = torch.matmul(m, self.generator_matrix) % self.q

        x_inner = self.marker_code.numba_encode(x.view(np.ndarray)).squeeze()
        x_inner = self.conv_code.encode(x_inner)

        M = (
            np.random.randint(self.n_sequence_min, self.n_sequence_max + 1)
            if self.fixed_n_sequence is None
            else self.fixed_n_sequence
        )

        r, N = self.channel.numba_transmit(x_inner, M)

        r, N = self.trellis.resize_received_sequence(r, N)

        d_min, d_max = self.trellis.evaluate_sequence_drift(
            N,
        )

        if self.compare_bcjr:

            dp, r = create_decoding_params(
                conv_code=self.conv_code,
                r=r.copy(),
                N=N.copy(),
                ids_channel=self.channel,
                max_insertions=self.max_insertions_per_symbol,
                std_mult=self.std_mult,
            )

            L_ei = self.conv_code.full_numba_bcjr_ids(r, self.L_ai, dp, self.channel)
            # normalize and transfer to probability space
            L_ei = np.exp(np.apply_along_axis(self.linear_code.log_normalize, 1, L_ei))
            L_ei = self.marker_code.decode(L_ei)
            x_pred_bcjr = L_ei.argmax(axis=1)[: self.linear_code.n]

        else:
            x_pred_bcjr = torch.full((self.linear_code.n,), -1).long()

        y = self.trellis.get_windowed_version(
            r.astype(int), N, d_min, d_max, self.conv_code.offset
        )

        y = np.pad(
            y,
            ((0, self.n_sequence_max - M), (0, 0), (0, 0), (0, 0)),
            constant_values=self.padding_value,
        )

        padding_mask = y[:, :, 0, 0] == self.padding_value

        return (
            torch.from_numpy(m).float(),
            torch.from_numpy(x.squeeze()).float(),
            torch.tensor(x_inner).float(),
            x_pred_bcjr,
            torch.tensor(y).float(),
            torch.tensor(padding_mask),
        )


class BCJRFormerConvDataset(data.Dataset):
    def __init__(
        self,
        linear_code: LinearCode,
        conv_code: LinearConvolutionalCode,
        p_i: float,
        p_d: float,
        p_s: float,
        batch_size: int,
        batches_per_epoch: int,
        compare_bcjr: bool = True,
        std_mult: float = 3.5,
        n_sequence_min=1,
        n_sequence_max=1,
        fixed_n_sequence: int | None = None,
        q: int = 2,
    ):
        self.linear_code = linear_code

        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size

        self.conv_code = conv_code

        self.q = q

        # Use python calculate because otherwise we get issues with their parallel matmul within a parallel dataloader setup
        self.GF = galois.GF(self.q, compile="python-calculate")
        self.generator_matrix = self.GF(linear_code.generator_matrix)
        self.pc_matrix = self.GF(linear_code.pc_matrix)

        channel = IDSChannel(
            q=2,
            p_i=p_i,
            p_d=p_d,
            p_s=p_s,
        )
        self.max_insertions_per_symbol = 2

        # Prior log likelihoods for convolutional code (everything has p=0.5)
        # Here we ignore relations between convolutional bits caused by the inner decoder
        # => We can add this using cross attention if we want to
        L_ai_bit = np.zeros((self.conv_code.encoded_length, 2))
        L_ai_bit[:, 1] = np.log(1 / linear_code.q)
        L_ai_bit[:, 0] = np.log(1 / linear_code.q)
        self.L_ai_bit = L_ai_bit

        # self.L_ai_state = self.conv_code.get_prior_information()

        self.channel = channel

        # TODO: Put in config if necessary
        self.compare_bcjr = compare_bcjr

        self.std_mult = std_mult
        self.n_sequence_min = n_sequence_min
        self.n_sequence_max = n_sequence_max

        self.trellis = BCJRFormerTrellis(
            channel=channel,
            T=self.conv_code.encoded_length,
            n_v=1,
            I_max=self.max_insertions_per_symbol,
            prior_llrs=self.L_ai_bit,
            std_mult=self.std_mult,
        )

        self.window_block_dimension = self.trellis.get_block_size()

        self.padding_value = -1

        self.fixed_n_sequence = fixed_n_sequence

    def __len__(self):
        # infinite dataset basically
        return int(self.batches_per_epoch * self.batch_size)

    def __getitem__(self, index):

        M = (
            np.random.randint(self.n_sequence_min, self.n_sequence_max + 1)
            if self.fixed_n_sequence is None
            else self.fixed_n_sequence
        )

        m = self.GF.Random((1, self.linear_code.k))

        x = m @ self.generator_matrix

        x_inner = self.conv_code.encode(x.squeeze()).squeeze()

        r, N = self.channel.numba_transmit(x_inner, M)

        r, N = self.trellis.resize_received_sequence(r, N)

        d_min, d_max = self.trellis.evaluate_sequence_drift(
            N,
        )

        # Just to keep thigns simple we don't perform BCJR here
        x_pred_bcjr = torch.full((self.linear_code.n,), -1).long()

        y = self.trellis.get_windowed_version(r, N, d_min, d_max, self.conv_code.offset)

        y = np.pad(
            y,
            ((0, self.n_sequence_max - M), (0, 0), (0, 0), (0, 0)),
            constant_values=self.padding_value,
        )
        padding_mask = y[:, :, 0, 0] == self.padding_value

        return (
            torch.from_numpy(m).float(),
            torch.from_numpy(x).squeeze().float(),
            torch.tensor(x_inner).float(),
            x_pred_bcjr,
            torch.tensor(y).float(),
            torch.tensor(padding_mask),
        )


class BCJRFormerConvStateDataset(data.Dataset):
    def __init__(
        self,
        linear_code: LinearCode,
        conv_code: LinearConvolutionalCode,
        p_i: float,
        p_d: float,
        p_s: float,
        batch_size: int,
        batches_per_epoch: int,
        compare_bcjr: bool = True,
        std_mult: float = 3.5,
        n_sequence_min=1,
        n_sequence_max=1,
        fixed_n_sequence: int | None = None,
    ):
        self.linear_code = linear_code

        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size

        self.generator_matrix = linear_code.generator_matrix
        self.pc_matrix = linear_code.pc_matrix

        self.conv_code = conv_code

        channel = IDSChannel(
            q=2,
            p_i=p_i,
            p_d=p_d,
            p_s=p_s,
        )
        self.max_insertions_per_symbol = 2

        # Prior log likelihoods for convolutional code (everything has p=0.5)
        # Here we ignore relations between convolutional bits caused by the inner decoder
        # => We can add this using cross attention if we want to

        # self.L_ai_state = self.conv_code.get_prior_information()

        self.channel = channel

        self.compare_bcjr = compare_bcjr

        L_ai_state = np.full(
            (self.conv_code.B, self.linear_code.q**self.conv_code.n_v),
            np.log(1 / (self.linear_code.q**self.conv_code.n_v)),
        )

        self.std_mult = std_mult
        self.n_sequence_min = n_sequence_min
        self.n_sequence_max = n_sequence_max

        self.trellis = BCJRFormerTrellis(
            channel=channel,
            T=self.conv_code.B,
            n_v=self.conv_code.n_v,
            I_max=self.max_insertions_per_symbol,
            prior_llrs=L_ai_state,
            std_mult=self.std_mult,
        )

        self.window_block_dimension = self.trellis.get_block_size()

        self.padding_value = -1

        self.fixed_n_sequence = fixed_n_sequence

    def __len__(self):
        # infinite dataset basically
        return int(self.batches_per_epoch * self.batch_size)

    def __getitem__(self, index):
        M = (
            np.random.randint(self.n_sequence_min, self.n_sequence_max + 1)
            if self.fixed_n_sequence is None
            else self.fixed_n_sequence
        )

        m = np.random.randint(0, 2, (1, self.linear_code.k))
        x = m @ self.generator_matrix % 2

        x_inner = self.conv_code.encode(x.squeeze()).squeeze()

        r, N = self.channel.numba_transmit(x_inner, M)

        r, N = self.trellis.resize_received_sequence(r, N)

        d_min, d_max = self.trellis.evaluate_sequence_drift(
            N,
        )

        # Just to keep thigns simple we don't perform BCJR here
        x_pred_bcjr = torch.full((self.linear_code.n,), -1).long()

        y = self.trellis.get_windowed_version(r, N, d_min, d_max, self.conv_code.offset)

        x_target = torch.cat(
            (torch.from_numpy(x).squeeze(), torch.tensor([0] * self.conv_code.m))
        )
        y = np.pad(
            y,
            ((0, self.n_sequence_max - M), (0, 0), (0, 0), (0, 0)),
            constant_values=self.padding_value,
        )

        padding_mask = y[:, :, 0, 0] == self.padding_value

        return (
            torch.from_numpy(m).float(),
            torch.from_numpy(x).squeeze().float(),
            x_target.float(),
            x_pred_bcjr,
            torch.tensor(y).float(),
            torch.tensor(padding_mask),
        )


class BCJRFormerConvCombinedDataset(data.Dataset):
    def __init__(
        self,
        linear_code: LinearCode,
        conv_code: LinearConvolutionalCode,
        p_i: float,
        p_d: float,
        p_s: float,
        batch_size: int,
        batches_per_epoch: int,
        compare_bcjr: bool = True,
        std_mult: float = 3.5,
        n_sequence_min=1,
        n_sequence_max=1,
        fixed_n_sequence: int | None = None,
    ):
        self.linear_code = linear_code

        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size

        self.generator_matrix = linear_code.generator_matrix
        self.pc_matrix = linear_code.pc_matrix

        self.conv_code = conv_code

        channel = IDSChannel(
            q=2,
            p_i=p_i,
            p_d=p_d,
            p_s=p_s,
        )
        self.max_insertions_per_symbol = 2

        # Prior log likelihoods for convolutional code (everything has p=0.5)
        # Here we ignore relations between convolutional bits caused by the inner decoder
        # => We can add this using cross attention if we want to

        # self.L_ai_state = self.conv_code.get_prior_information()

        self.channel = channel

        self.compare_bcjr = compare_bcjr

        L_ai_state = np.full(
            (self.conv_code.B, self.linear_code.q**self.conv_code.n_v),
            np.log(1 / (self.linear_code.q**self.conv_code.n_v)),
        )

        L_ai_bit = np.zeros((self.conv_code.encoded_length, 2))
        L_ai_bit[:, 1] = np.log(1 / linear_code.q)
        L_ai_bit[:, 0] = np.log(1 / linear_code.q)

        self.std_mult = std_mult
        self.n_sequence_min = n_sequence_min
        self.n_sequence_max = n_sequence_max

        self.state_trellis = BCJRFormerTrellis(
            channel=channel,
            T=self.conv_code.B,
            n_v=self.conv_code.n_v,
            I_max=self.max_insertions_per_symbol,
            prior_llrs=L_ai_state,
            std_mult=self.std_mult,
            full_state=True,
        )

        self.bit_trellis = BCJRFormerTrellis(
            channel=channel,
            T=self.conv_code.encoded_length,
            n_v=1,
            I_max=self.max_insertions_per_symbol,
            prior_llrs=L_ai_bit,
            std_mult=self.std_mult,
            full_state=True,
        )

        self.bit_window_block_dimension = self.bit_trellis.get_block_size()
        self.state_window_block_dimension = self.state_trellis.get_block_size()

        self.padding_value = -1

        self.fixed_n_sequence = fixed_n_sequence

    def __len__(self):
        # infinite dataset basically
        return int(self.batches_per_epoch * self.batch_size)

    def __getitem__(self, index):
        M = (
            np.random.randint(self.n_sequence_min, self.n_sequence_max + 1)
            if self.fixed_n_sequence is None
            else self.fixed_n_sequence
        )

        m = np.random.randint(0, 2, (1, self.linear_code.k))
        x = m @ self.generator_matrix % 2

        if self.conv_code.offset.any():
            self.conv_code.new_random_offset()

        x_inner = self.conv_code.encode(x.squeeze()).squeeze()

        x_inner_no_offset = (x_inner + self.conv_code.offset) % (1 << self.conv_code.p)

        r, N = self.channel.numba_transmit(x_inner, M)

        _, N_state = self.state_trellis.resize_received_sequence(r.copy(), N)
        _, N_bit = self.bit_trellis.resize_received_sequence(r.copy(), N)

        d_min_state, d_max_state = self.state_trellis.evaluate_sequence_drift(
            N_state,
        )

        d_min_bit, d_max_bit = self.bit_trellis.evaluate_sequence_drift(
            N_bit,
        )

        # Just to keep thigns simple we don't perform BCJR here
        x_pred_bcjr = torch.full((self.linear_code.n,), -1).long()

        y_bit = self.bit_trellis.get_windowed_version(
            r, N_bit, d_min_bit, d_max_bit, self.conv_code.offset
        )

        y_state = self.state_trellis.get_windowed_version(
            r, N_state, d_min_state, d_max_state, self.conv_code.offset
        )

        x_outer_target = torch.cat(
            (torch.from_numpy(x).squeeze(), torch.tensor([0] * self.conv_code.m))
        )
        y_bit = np.pad(
            y_bit,
            ((0, self.n_sequence_max - M), (0, 0), (0, 0), (0, 0)),
            constant_values=self.padding_value,
        )

        y_state = np.pad(
            y_state,
            ((0, self.n_sequence_max - M), (0, 0), (0, 0), (0, 0)),
            constant_values=self.padding_value,
        )

        padding_mask = np.concatenate(
            (
                y_bit[:, :, 0, 0] == self.padding_value,
                y_state[:, :, 0, 0] == self.padding_value,
            ),
            axis=1,
        )

        return (
            torch.from_numpy(m).float(),
            torch.from_numpy(x).squeeze().float(),
            torch.from_numpy(x_inner_no_offset).squeeze().float(),
            x_outer_target.float(),
            x_pred_bcjr,
            torch.tensor(y_bit).float(),
            torch.tensor(y_state).float(),
            torch.tensor(padding_mask),
        )
