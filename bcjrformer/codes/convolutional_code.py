from typing import Callable, Protocol
from bcjrformer.channels import IDSChannel
from bcjrformer.codes.decoding_params import (
    DecodingParameters,
)

from numpy.typing import NDArray
import numba as nb
import numpy as np
from numba import njit

from numba.extending import intrinsic
from numba.cpython import mathimpl

from numba import types

import math

from bcjrformer.utils import calculate_standard_form_bin


@njit(cache=True)
def dec_to_vector(i: int, v: np.ndarray, size: int, q: int):
    for j in range(size):
        v[j] = i % q
        i //= q


@njit(cache=True)
def vector_to_dec(v: np.ndarray, q: int) -> int:
    bit_multiplier = 1
    dec = 0
    for v_j in v[::-1]:
        dec += v_j * bit_multiplier
        bit_multiplier *= q
    return dec


# From: https://stackoverflow.com/a/77103233
@intrinsic
def popcnt(typingctx, src):
    sig = types.uint64(types.uint64)

    def codegen(context, builder, signature, args):
        return mathimpl.call_fp_intrinsic(builder, "llvm.ctpop.i64", args)

    return sig, codegen


@njit(cache=True)
def decimal_to_drift(
    dec: int,
    drift: np.ndarray,
    A: int,
    t: int,
    dp_d_min: np.ndarray,
    dp_d_max: np.ndarray,
):
    D = dp_d_max[:, t] - dp_d_min[:, t] + 1
    for j in range(A):
        drift[j] = dec % D[j] + dp_d_min[j, t]
        dec //= D[j]


@njit(cache=True)
def decimal_to_relative_drift(
    dec: int,
    drift: np.ndarray,
    A: int,
    d_min: int,
    d_max: int,
):
    D = d_max - d_min + 1
    for j in range(A):
        drift[j] = dec % D + d_min
        dec //= D


@njit(cache=True)
def drift_to_decimal(
    drift: np.ndarray,
    t: int,
    A: int,
    dp_d_min: np.ndarray,
    dp_d_max: np.ndarray,
) -> int:
    dec = 0
    b = 1
    for j in range(A):
        dec += (drift[j] - dp_d_min[j, t]) * b
        b *= dp_d_max[j, t] - dp_d_min[j, t] + 1

    return dec


@njit(cache=True)
def drift_to_decimal_vector(
    drift: np.ndarray,
    t: int,
    A: int,
    dp_d_min: np.ndarray,
    dp_d_max: np.ndarray,
) -> np.ndarray:
    multipliers = np.ones(A, dtype=np.int64)
    for j in range(1, A):
        multipliers[j] = multipliers[j - 1] * (
            dp_d_max[j - 1, t] - dp_d_min[j - 1, t] + 1
        )
    diffs = drift - dp_d_min[:, t]

    result = (diffs * multipliers).sum(axis=1)

    return result


def _numba_all_last_axis(A: int) -> Callable[[np.ndarray], np.ndarray]:
    """Function to perform .all() along the last axis of an array

    Args:
        A (int): The number of axes

    Returns:
        Callable[[np.ndarray], np.ndarray]: A numba compiled function to perform .all() along the last axis
    """
    if A == 1:

        def _all_axis(arr: np.ndarray) -> np.ndarray:
            return arr[..., 0]

    else:

        def _all_axis(arr: np.ndarray) -> np.ndarray:
            out = np.logical_and(arr[..., 0], arr[..., 1])
            for ix in range(2, A):
                out = np.logical_and(out, arr[..., ix])
            return out

    return nb.njit(_all_axis, cache=True)


class BCJRIDSChannel:
    q_i: int
    "Number of codewords per trellis section"

    n_v: int  # could be extended to be an array eventually if MRs are used
    "Number of output symbols per trellis section"

    cum_N: np.ndarray
    "The cumulative number of output symbols per trellis section"

    B: int
    "The number of time steps to encode the message, i. e. total T + m"

    encoded_length: int
    "The length of the encoded message"

    m: int
    "The memory of the Code"

    num_states: int
    "The number of states in the trellis"

    _numba_get_o_LD: None | nb.types.FunctionType = None
    _numba_bcjr_ids: None | Callable = None
    _numba_encode: None | Callable[[np.ndarray, np.ndarray], np.ndarray] = None

    get_output_label: Callable[[int, int, int], NDArray[np.float64]]
    """A function that given a timestep, current state and input symbol returns the output label"""

    get_out_edg: Callable[[int, int], int]
    """A function that given a state and input symbol returns the next state"""

    p: int
    """2**p is the number of bits for the offset"""

    def __init__(
        self,
        get_output_label: Callable[[int, int, int], np.ndarray],
        get_out_edg: Callable[[int, int], int],
        q_i: int,
        n_v: int,
        B: int,
        encoded_length: int,
        cum_N: np.ndarray,
        num_states: int,
        m: int,
        p: int,
    ):
        self.get_output_label = get_output_label
        self.get_out_edg = get_out_edg
        self.q_i = q_i
        self.n_v = n_v
        self.B = B
        self.encoded_length = encoded_length
        self.cum_N = cum_N
        self.num_states = num_states
        self.m = m

        self.o_E: np.ndarray = np.fromfunction(
            lambda _, s, o: get_out_edg(s, o),
            (self.B, self.num_states, self.q_i),
            dtype=np.int32,
        )

        self.p = p

    def full_numba_bcjr_ids(
        self,
        r: np.ndarray,
        L_ai: np.ndarray,
        dp: "DecodingParameters",
        ids_channel: IDSChannel,
        offset: np.ndarray,
    ):
        if self._numba_bcjr_ids is not None:
            return self._numba_bcjr_ids(
                r, dp.num_drift_states, dp.d_min, dp.d_max, dp.max_N, dp.N, offset
            )

        get_out_lab = self.get_output_label

        L_ai = L_ai.copy()

        ids_q = ids_channel.q
        ids_p_i = ids_channel.p_i
        ids_p_d = ids_channel.p_d
        ids_p_s = ids_channel.p_s
        ids_p_t = ids_channel.p_t
        o_E = self.o_E
        dp_num_relative_drift_states = dp.num_relative_drift_states
        dp_max_insertions = dp.max_insertions
        dp_A = dp.A
        dp_max_Y_length = dp.max_Y_length
        cc_B = self.B
        cc_nv = self.n_v
        cc_q_i = self.q_i
        cc_p = self.p
        cc_cum_N = self.cum_N
        cc_num_states = self.num_states
        cc_encoded_length = self.encoded_length
        numba_all_last_axis = _numba_all_last_axis(dp.A)

        @njit
        def _numba_bcjr_ids(
            rec: np.ndarray,
            dp_num_drift_states: np.ndarray,
            dp_d_min: np.ndarray,
            dp_d_max: np.ndarray,
            dp_maxN: int,
            dp_N: np.ndarray,
            cc_offset: np.ndarray,
        ):
            rel_Dv = np.zeros((dp_num_relative_drift_states, dp_A), dtype=np.int16)

            for d_ in range(dp_num_relative_drift_states):
                decimal_to_relative_drift(
                    d_, rel_Dv[d_], dp_A, -cc_nv, cc_nv * dp_max_insertions
                )
            Dv = np.empty((cc_B, dp_num_drift_states.max(), dp_A), dtype=np.int16)
            num_valid_relative_drift_states = np.zeros(
                (cc_B, dp_num_drift_states.max()), dtype=np.int16
            )

            valid_rel_Dv = np.empty(
                (cc_B, dp_num_drift_states.max(), rel_Dv.shape[0], dp_A),
                dtype=np.int16,
            )

            valid_rel_Dd = np.empty(
                (cc_B, dp_num_drift_states.max(), rel_Dv.shape[0]),
                dtype=np.int16,
            )

            for t in range(cc_B):
                for d in range(dp_num_drift_states[t]):
                    decimal_to_drift(d, Dv[t, d], dp_A, t, dp_d_min, dp_d_max)

                    d_rel_Dv = Dv[t, d] + rel_Dv

                    valid_d_rel_Dv_mask = numba_all_last_axis(
                        (d_rel_Dv >= dp_d_min[:, t + 1])
                    ) & numba_all_last_axis(d_rel_Dv <= dp_d_max[:, t + 1])

                    num_valid_relative_drift_states[t, d] = valid_d_rel_Dv_mask.sum()

                    valid_d_rel_Dv = d_rel_Dv[valid_d_rel_Dv_mask]

                    valid_d_rel_Dd = drift_to_decimal_vector(
                        valid_d_rel_Dv, t + 1, dp_A, dp_d_min, dp_d_max
                    )

                    valid_rel_Dd[t, d, : num_valid_relative_drift_states[t, d]] = (
                        valid_d_rel_Dd
                    )

                    valid_rel_Dv[t, d, : num_valid_relative_drift_states[t, d]] = (
                        valid_d_rel_Dv
                    )
            F = np.zeros(
                (
                    dp_A,
                    dp_max_Y_length + 1,
                    int(math.pow(ids_q, cc_nv)),
                    dp_maxN + 1,
                ),
                dtype=np.float64,
            )

            xv = np.empty(cc_nv, dtype=np.int16)
            y = np.empty(dp_max_Y_length, dtype=np.int8)
            xt: int

            F_out = np.zeros((cc_nv + 1, dp_max_Y_length + 1))

            for j in range(dp_A):
                N_j = dp_N[j]
                r_j = rec[j, :N_j]
                for i in range(N_j + 1):
                    for x in range(int(math.pow(ids_q, cc_nv))):
                        if i + dp_max_Y_length <= N_j:
                            y = r_j[i : i + dp_max_Y_length]
                        else:
                            y = np.concatenate(
                                (
                                    r_j[i:],
                                    np.zeros(
                                        dp_max_Y_length - N_j + i,
                                        dtype=np.int8,
                                    ),
                                )
                            )

                        Ft = np.zeros((cc_nv + 1, dp_max_Y_length + 1))

                        dec_to_vector(x, xv, cc_nv, ids_q)

                        Ft[0, 0] = 1

                        for v in range(dp_max_Y_length + 1):
                            for p in range(cc_nv + 1):
                                # deletions
                                if p > 0:
                                    Ft[p, v] += ids_p_d * Ft[p - 1][v]

                                # substitutions
                                if p > 0 and v > 0:
                                    Ft[p, v] += Ft[p - 1, v - 1] * (
                                        (ids_p_t * (1 - ids_p_s))
                                        if xv[p - 1] == y[v - 1]
                                        else (ids_p_t * ids_p_s) / (ids_q - 1)
                                    )

                                # Note that we are aiming to get the output probability not transition probability
                                # So we need to exclude insertions for the output probability
                                F_out[p, v] = Ft[p, v]

                                # insertions
                                if v > 0:
                                    Ft[p][v] += ids_p_i * Ft[p][v - 1] / ids_q

                        xt = vector_to_dec(xv, ids_q)

                        F[j, :, xt, i] = np.log(F_out[cc_nv, :])

            o_LD = np.zeros((cc_B, cc_num_states, cc_q_i), np.int16)

            # # Precompute powers of alph_q:
            # E. g. for q=4 and nv = 2: (4^1, 4^0)
            alphabet_powers = (1 << cc_p) ** np.arange(
                cc_nv - 1, -1, -1, dtype=np.float64
            )
            for t in range(cc_B):
                for s in range(cc_num_states):
                    for m_i in range(cc_q_i):

                        labels = get_out_lab(t, s, m_i)

                        # With offset
                        offset_label = (
                            labels + cc_offset[t * cc_nv : (t + 1) * cc_nv]
                        ) % (1 << cc_p)

                        # Integer representation of the concatenated integer vector
                        # examples: binary: [1, 0] -> 2; quarternay: [3, 1] -> 13
                        o_LD[t, s, m_i] = int(np.dot(offset_label, alphabet_powers))

            alpha = np.full(
                (cc_B + 1, cc_num_states, dp_num_drift_states.max()),
                -np.inf,
                dtype=np.float64,
            )
            alpha[0][0][0] = 0
            dv = dp_N - cc_encoded_length

            beta = np.full_like(alpha, -np.inf)
            beta[cc_B][0][drift_to_decimal(dv, cc_B, dp_A, dp_d_min, dp_d_max)] = 0

            for t in range(cc_B):
                # Forward pass
                for d in range(dp_num_drift_states[t]):
                    dv = Dv[t, d]
                    for rel_dix in range(num_valid_relative_drift_states[t, d]):
                        dd_ = valid_rel_Dd[t, d, rel_dix]
                        dv_ = valid_rel_Dv[t, d, rel_dix]

                        for s in range(cc_num_states):
                            for o in range(cc_q_i):
                                G = 0
                                for j in range(dp_A):
                                    G += F[
                                        j,
                                        dv_[j]
                                        - dv[j]
                                        + cc_nv,  # valid_rel_Dv - d_rel_Dv[:, valid_rel_Dv_to_valid_Dv] + cc_nv
                                        o_LD[t][s][o],
                                        cc_cum_N[t] + dv[j],
                                    ]

                                alpha[t + 1][o_E[t][s][o]][dd_] = np.logaddexp(
                                    alpha[t + 1][o_E[t][s][o]][dd_],
                                    alpha[t][s][d] + G + L_ai[t][o],
                                )

                # Backward pass
                t_beta = cc_B - t - 1
                for d in range(dp_num_drift_states[t_beta]):
                    dv = Dv[t_beta, d]
                    for rel_dix in range(num_valid_relative_drift_states[t_beta, d]):
                        dd_ = valid_rel_Dd[t_beta, d, rel_dix]
                        dv_ = valid_rel_Dv[t_beta, d, rel_dix]

                        for s in range(cc_num_states):
                            for o in range(cc_q_i):
                                G = 0
                                for j in range(dp_A):
                                    G += F[
                                        j,
                                        dv_[j] - dv[j] + cc_nv,
                                        o_LD[t_beta][s][o],
                                        cc_cum_N[t_beta] + dv[j],
                                    ]

                                beta[t_beta][s][d] = np.logaddexp(
                                    beta[t_beta][s][d],
                                    beta[t_beta + 1][o_E[t_beta][s][o]][dd_]
                                    + G
                                    + L_ai[t_beta][o],
                                )

            # LLR calculation
            L_e = np.full((cc_B, cc_q_i), -np.inf)
            for t in range(cc_B - 1, -1, -1):
                for d in range(dp_num_drift_states[t]):
                    dv = Dv[t, d]
                    for rel_dix in range(num_valid_relative_drift_states[t, d]):
                        dd_ = valid_rel_Dd[t, d, rel_dix]
                        dv_ = valid_rel_Dv[t, d, rel_dix]
                        for s in range(cc_num_states):
                            for o in range(cc_q_i):
                                G = 0
                                for j in range(dp_A):
                                    G += F[
                                        j,
                                        dv_[j] - dv[j] + cc_nv,
                                        o_LD[t][s][o],
                                        cc_cum_N[t] + dv[j],
                                    ]

                                L_e[t][o] = np.logaddexp(
                                    L_e[t][o],
                                    alpha[t][s][d]
                                    + beta[t + 1][o_E[t][s][o]][dd_]
                                    + G
                                    + L_ai[t][o],
                                )

            return L_e

        self._numba_bcjr_ids = _numba_bcjr_ids

        return self._numba_bcjr_ids(
            r, dp.num_drift_states, dp.d_min, dp.d_max, dp.max_N, dp.N, offset
        )

    def encode(self, message: np.ndarray, offset: np.ndarray) -> np.ndarray:
        if self._numba_encode is None:
            self._numba_encode = self.numba_compile_encode()

        return self._numba_encode(message, offset)

    def numba_compile_encode(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        encoded_length = self.encoded_length
        m = self.m
        n_v = self.n_v
        get_out_edg = self.get_out_edg
        get_output_label = self.get_output_label
        p = self.p

        @njit
        def _encode(message: np.ndarray, offset: np.ndarray):

            enc = np.zeros(encoded_length, dtype=np.int64)

            m_pad = np.concatenate((message, np.zeros(m, dtype=np.int64)))

            s = 0
            for t, m_i in enumerate(m_pad):

                enc[t * n_v : (t + 1) * n_v] = (
                    get_output_label(t, s, m_i) + offset[t * n_v : (t + 1) * n_v]
                ) % (1 << p)
                s = get_out_edg(s, m_i)

            return enc

        return _encode


class ConvolutionalCode(Protocol):
    # Required attributes
    p: int
    T: int
    B: int
    encoded_length: int
    num_states: int
    n_v: int
    q_i: int
    cum_N: np.ndarray
    offset: np.ndarray
    get_output_label: Callable[[int, int, int], NDArray[np.float64]]
    get_out_edg: Callable[[int, int], int]
    ids_bcjr: BCJRIDSChannel

    # Optional attributes (use Optional for attributes that can be None)
    _numba_get_o_LD: nb.types.FunctionType | None
    _full_numba_bcjr_ids: Callable | None

    # Required methods
    def encode(self, message: np.ndarray) -> np.ndarray: ...

    def full_numba_bcjr_ids(
        self,
        r: np.ndarray,
        L_ai: np.ndarray,
        dp: "DecodingParameters",
        ids_channel: "IDSChannel",
    ) -> np.ndarray: ...

    def new_random_offset(self) -> None: ...

    def _numba_compile_get_output_label(
        self,
    ) -> Callable[[int, int, int], NDArray[np.float64]]: ...

    def _numba_compile_get_out_edg(self) -> Callable[[int, int], int]: ...


class LinearConvolutionalCode:
    """Class to interact with Convolutional Codes"""

    k: int
    "Convolutional Code k (Number of input bits)"

    g: list[int]
    "The generator polynomials of the Code. Example: [7, 5], equivalent to 111 and 101"

    p: int
    """Should be the the number of bits to encode the alphabet (e. g. 2 bits if
    q = 4, 3 bits if q = 8, ...)
    """

    cum_N: np.ndarray
    "The cumulative number of output symbols per trellis section"

    q_i: int
    "Number of codewords per trellis section"

    n_v: int
    "Number of output symbols per trellis section"

    n: int
    "Number of output symbols per trellis section"

    T: int
    "The number of time steps of the message"

    B: int
    "The number of time steps to encode the message, i. e. total T + m"

    encoded_length: int
    "The length of the encoded message"

    _numba_get_o_LD: None | nb.types.FunctionType = None
    _full_numba_bcjr_ids: None | Callable = None

    _generator_matrix: None | np.ndarray = None
    _pc_matrix: None | np.ndarray = None
    _tg_distance_matrix: None | np.ndarray = None

    def __init__(
        self, k: int, g: list[int], p: int, T: int, random_offset: bool = True
    ):
        """Constructor for ConvolutionalCode

        Args:
            k (int): Convolutional Code k
            g (list[int]): Convolutional Code g
            p (int): Should be the the number of bits to encode the alphabet
            (e. g. 2 bits if q = 4, 3 bits if q = 8, ...)
            T (int): The number of time steps to encode the message
            (i. e. the number of bits in the input message)
            random_offset (bool, optioget_out_edgnal): If True, the offset will be random.
        """

        self.k = k
        self.g = g
        self.p = p

        assert len(g) % p == 0, "The length of g must be divisible by p"

        self.n = int(len(g) / p)

        self.m = self.calculate_memory()

        self.T = T

        self.B = T + self.m

        if random_offset:
            self.offset = np.random.randint(0, 1 << p, self.B * self.n)
        else:
            self.offset = np.zeros(self.B * self.n, dtype=np.int16)

        self.num_states = 1 << (k * self.m)

        self.n_v = self.n

        self.q_i = 1 << self.k

        # self.cum_N = np.cumsum([0] + [self.n_v] * self.B)
        self.cum_N = np.arange(0, self.B * self.n_v + 1, self.n_v)
        self.encoded_length = self.cum_N[self.B]

        self.get_output_label = self._numba_compile_get_output_label()

        self.get_out_edg = self._numba_compile_get_out_edg()

        self.L_ai = self.get_prior_likelihoods()

        self.ids_bcjr = BCJRIDSChannel(
            self.get_output_label,
            self.get_out_edg,
            self.q_i,
            self.n_v,
            self.B,
            self.encoded_length,
            self.cum_N,
            self.num_states,
            self.m,
            self.p,
        )

    def encode(self, message: np.ndarray) -> np.ndarray:
        return self.ids_bcjr.encode(message, self.offset)

    def get_prior_likelihoods(self) -> np.ndarray:
        # A priori input bit information
        L_ai = np.zeros((self.B, (1 << self.p)))

        # From the Convolutional Code we know that the last m bits are zero => first bit is log(1) = 0 and others are -inf
        L_ai[self.T :, 1:] = -np.inf

        # The first n bits have even a priori probabilities between all q alphabet symbols
        L_ai[: self.T, :] = np.log(1 / (1 << self.p))

        return L_ai

    def full_numba_bcjr_ids(
        self,
        r: np.ndarray,
        L_ai: np.ndarray,
        dp: "DecodingParameters",
        ids_channel: "IDSChannel",
    ):
        return self.ids_bcjr.full_numba_bcjr_ids(r, L_ai, dp, ids_channel, self.offset)

    def get_generator_matrix(self) -> np.ndarray:

        # we extend the generator matrix initially to account for the first states in the shift register
        ext_generator_matrix = np.zeros(
            (self.encoded_length, self.B + self.m), dtype=int
        )

        if self.p > 2:
            raise NotImplementedError("Generator matrix only supported for p = 2")
        generator_block = np.zeros((self.n, self.m + 1), dtype=int)
        for ix, poly in enumerate(self.g):
            vec = np.zeros(self.m + 1, dtype=int)
            dec_to_vector(poly, vec, self.m + 1, 1 << self.p)
            generator_block[ix] = vec[::-1]

        for t in range(self.B):
            ext_generator_matrix[t * self.n : (t + 1) * self.n, t : t + self.m + 1] = (
                generator_block
            )

        generator_matrix = ext_generator_matrix[:, self.m :].T

        return generator_matrix

    def get_pc_matrix(self) -> np.ndarray:
        if self.p > 2:
            raise NotImplementedError("Parity Check matrix only supported for p = 2")

        generator_matrix = self.generator_matrix

        standard_form_generator_matrix = calculate_standard_form_bin(
            generator_matrix, self.B
        )
        pc_matrix = np.concatenate(
            [
                standard_form_generator_matrix[:, self.B :].T,
                np.eye(self.B, dtype=np.int64),
            ],
            axis=1,
        )

        return pc_matrix

    @property
    def generator_matrix(self) -> np.ndarray:
        if self._generator_matrix is None:
            self._generator_matrix = self.get_generator_matrix()
        return self._generator_matrix

    @property
    def pc_matrix(self) -> np.ndarray:
        if self._pc_matrix is None:
            self._pc_matrix = self.get_pc_matrix()
        return self._pc_matrix

    def new_random_offset(self):
        self.offset = np.random.randint(0, 1 << self.p, self.B * self.n)

    def calculate_memory(self) -> int:
        """Calculates the memory of the Convolutional Code

        The memory of a convolutional code is the maximum number of bits that can be
        present in the shift register at any time.

        Returns:
            int: The memory of the Convolutional Code
        """
        memory = 0
        for g_i in self.g:
            while g_i >> (self.k * (memory + 1)):
                memory += 1
        return memory

    def _numba_compile_get_output_label(self) -> Callable[[int, int, int], np.ndarray]:

        k = self.k
        m = self.m
        g = np.array(self.g, dtype=np.int16)
        p = self.p
        n_v = self.n_v

        popcount = self.popcount

        @njit
        def _get_output_label(t, s, m_i) -> np.ndarray:
            # Note that for a regular convolutional code, the output label is independent of the time t
            # (though the last timestep(s) are encoded due the state of the shift register)

            labels = np.zeros((n_v,))

            shifted_state_mask = (m_i << (k * m)) | s

            for nn in range(n_v):
                for pp in range(p):
                    labels[nn] += (
                        popcount(shifted_state_mask & g[nn * p + pp]) % 2
                    ) << pp

            return labels

        return _get_output_label

    @staticmethod
    @nb.njit(nb.uint64(nb.uint64))
    def popcount(x):
        return popcnt(x)  # type: ignore

    def _numba_compile_get_out_edg(self) -> Callable[[int, int], int]:

        k = self.k
        m = self.m

        @njit
        def _get_out_edg(s, m_i) -> int:
            return (m_i << k * max(0, (m - 1))) | (s >> k)

        return _get_out_edg


class MarkerConvolutionalCode:
    """Class to interact with Convolutional Codes"""

    p: int
    """Should be the the number of bits to encode the alphabet (e. g. 2 bits if
    q = 4, 3 bits if q = 8, ...)
    """

    T: int
    "The input length (and output length, since its a marker code)"

    _numba_get_o_LD: None | nb.types.FunctionType = None
    _full_numba_bcjr_ids: None | Callable = None

    def __init__(self, p: int, T: int, random_offset: bool = False):
        """Constructor for MarkerConvolutionalCode

        Args:
            p (int): Should be the the number of bits to encode the alphabet
            (e. g. 2 bits if q = 4, 3 bits if q = 8, ...)
            T (int): The number of time steps to encode the message
            (i. e. the number of bits in the input message)
            random_offset (bool, optioget_out_edgnal): If True, the offset will be random.
        """

        self.p = p
        self.T = T

        if random_offset:
            self.offset = np.random.randint(0, 1 << p, self.T)
        else:
            self.offset = np.zeros(self.T, dtype=int)

        self.get_output_label = self._numba_compile_get_output_label()

        self.get_out_edg = self._numba_compile_get_out_edg()

        self.q_i = 1 << p
        self.n_v = 1
        self.B = T
        self.encoded_length = T
        self.cum_N = np.arange(0, T + 1, 1)
        self.num_states = 1
        self.m = 0

        self.ids_bcjr = BCJRIDSChannel(
            self.get_output_label,
            self.get_out_edg,
            self.q_i,
            self.n_v,
            self.B,
            self.encoded_length,
            cum_N=self.cum_N,
            num_states=self.num_states,
            m=self.m,
            p=self.p,
        )

    def encode(self, message: np.ndarray) -> np.ndarray:
        return (message + self.offset) % (1 << self.p)

    def full_numba_bcjr_ids(
        self,
        r: np.ndarray,
        L_ai: np.ndarray,
        dp: "DecodingParameters",
        ids_channel: IDSChannel,
    ):
        return self.ids_bcjr.full_numba_bcjr_ids(r, L_ai, dp, ids_channel, self.offset)

    def new_random_offset(self):
        self.offset = np.random.randint(0, 1 << self.p, self.B)

    def _numba_compile_get_output_label(self) -> Callable[[int, int, int], np.ndarray]:
        @njit
        def _get_output_label(t, s, m_i) -> np.ndarray:
            # For a marker code we never have to consider the time step or the state, we just use the current message
            # return np.array([m_i], dtype=np.float64)
            return np.array([m_i], dtype=np.float64)

        return _get_output_label

    def _numba_compile_get_out_edg(self) -> Callable[[int, int], int]:
        @njit
        def _get_out_edg(s, m_i) -> int:
            # For marker codes the state is always 0 since there is no memory
            return s >> 0

        return _get_out_edg
