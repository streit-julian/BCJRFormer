from typing import Callable

import numba as nb
import numpy as np

from bcjrformer.channels import IDSChannel
from bcjrformer.codes.decoding_params import get_sequence_drift
from bcjrformer.codes.galois_field import GaloisField2m
from bcjrformer.codes.decoding_params import (
    truncate_or_extend_codeword,
)


def _get_window_size(drift_windows: np.ndarray) -> int:
    return (
        np.max(np.abs(drift_windows[1])) + np.max(np.abs(drift_windows[0])) + 1
    ).astype(np.int64)


class BCJRFormerTrellis:

    _numba_get_o_LD: None | nb.types.FunctionType = None
    _full_numba_bcjr_ids: None | Callable = None

    def __init__(
        self,
        channel: IDSChannel,
        T: int,
        n_v: int,
        I_max: int,
        prior_llrs: np.ndarray,
        std_mult: float = 3.5,
        q: int = 2,
        full_state: bool = False,  # Whether to discard the last state since the probabilities need to add up to 1
    ):
        self.p_s = channel.p_s
        self.q = q  # alphabet size

        self.T = T  # Trellis timesteps
        self.n_v = n_v  # Number of output symbols per timestep
        self.I_max = I_max  # Maximum number of insertions per timestep
        self.full_state = full_state

        self.n_s = self.q**self.n_v if full_state else self.q**self.n_v - 1

        self.prior_llrs = prior_llrs  # Prior llrs

        self.cum_N = np.arange(
            0, T * n_v + 1, n_v
        )  # Cumulative number of output symbols
        self.encoded_length = self.cum_N[-1]

        (
            self.channel_avg,
            self.channel_var,
        ) = channel.get_final_drift_statistics()  # Channel statistics
        self.std_mult = std_mult

        self.maximum_drift_range = self.std_mult * np.sqrt(
            self.channel_var * self.encoded_length
        )  # factor for maximum drift range

        self.drift_window = self.get_drift_windows()
        self.window_size = _get_window_size(self.drift_window)

        self.get_windowed_version: Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ] = self.numba_get_windowed_version_func()

    def get_block_size(self) -> int:
        return self.window_size * self.n_s

    def get_drift_windows(self) -> np.ndarray:
        max_drift = np.zeros((2, self.T))

        avg, maximum_drift_range = (
            self.channel_avg,
            self.maximum_drift_range,
        )

        max_drift[0] = np.maximum(
            self.cum_N[1:] * avg - maximum_drift_range,
            -self.cum_N[1:],
        ).astype(int)
        max_drift[1] = np.minimum(
            self.cum_N[1:] * avg + maximum_drift_range,
            self.cum_N[1:] * self.I_max,
        ).astype(int)

        return max_drift

    def resize_received_sequence(
        self,
        r,
        N,
    ) -> tuple[np.ndarray, np.ndarray]:
        return truncate_or_extend_codeword(
            r,
            N,
            self.encoded_length,
            self.channel_avg,
            self.maximum_drift_range,
        )

    def evaluate_sequence_drift(self, N: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the drift adapted to multiple received sequences

        Args:
            N (np.ndarray): The length of the individual received sequences

        Returns:
            tuple[np.ndarray, np.ndarray]: The minimum and maximum drift for each received sequence
        """

        d_min = np.zeros((len(N), self.T), dtype=int)
        d_max = np.zeros((len(N), self.T), dtype=int)
        for i, Ni in enumerate(N):
            di = get_sequence_drift(
                Ni,
                self.encoded_length,
                self.T,
                self.cum_N,
                self.channel_avg,
                self.std_mult * np.sqrt(self.channel_var * self.encoded_length),
                self.I_max,
            )

            d_min[i] = di[0, 1:]
            d_max[i] = di[1, 1:]
        return d_min, d_max

    def numba_get_windowed_version_func(
        self,
    ) -> Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]:
        """Returns the numba version of the get_windowed_version function"""
        T = self.T

        window_size = self.window_size

        center = np.max(np.abs(self.drift_window[0])).astype(int)

        prior_probabilities = np.exp(self.prior_llrs)

        # precompute state transitions
        n_v = self.n_v
        n_t = self.q**n_v
        n_s = self.n_s

        gf_add = GaloisField2m(q=self.q).add

        state_transition_matrix = np.zeros((n_t, n_t))
        for i in range(n_t):
            for j in range(n_t):
                bc_disagree = (i ^ j).bit_count()
                bc_agree = n_v - bc_disagree

                state_transition_matrix[i, j] = (self.p_s**bc_disagree) * (
                    (1 - self.p_s) ** bc_agree
                )

        @nb.njit
        def numba_get_windowed_version(
            r: np.ndarray,
            N: np.ndarray,
            d_min: np.ndarray,
            d_max: np.ndarray,
            offset: np.ndarray,  # TODO: We have to basically permute everything to align with offset
        ) -> np.ndarray:

            y = np.empty((len(r), T, window_size, n_s))

            for i, ri in enumerate(r):
                Ni = N[i]

                for ix in range(T):
                    # we need one less state than the total state count, since the probabilities need to add up to 1
                    # => The last state does not provide any additional information
                    # This would change if e. g. there is a prior state probability instead of a bitwise ap probability
                    window = np.zeros((window_size, n_s))

                    # Determine the range in rec to map to the window

                    window_min = (
                        center + d_min[i, ix] if ix + d_min[i, ix] > 0 else center - ix
                    )

                    window_max = (
                        center + d_max[i, ix]
                        if ix + d_max[i, ix] < Ni
                        else center + Ni - ix
                    ) + 1

                    window_range = range(
                        window_min,
                        window_max,
                    )

                    rec_range = range(
                        max(ix + d_min[i, ix], 0),
                        min(ix + d_max[i, ix] + 1, Ni),
                    )

                    assert len(window_range) == len(
                        rec_range
                    ), "Window and rec range mismatch"

                    for wix, rix in zip(window_range, rec_range):
                        # Calculate the state bits for the current window
                        # => We need to add the offset here, to get the correct state
                        rec_state_bits = ri[rix : rix + n_v]

                        rec_state_bits = gf_add(rec_state_bits, offset[ix : ix + n_v])

                        rec_state_int = 0
                        for bit in rec_state_bits:
                            rec_state_int = (rec_state_int << 1) | bit

                        window[wix] = (
                            prior_probabilities[ix, :n_s]
                            * state_transition_matrix[:n_s, rec_state_int]
                        )
                    y[i, ix] = window

            return y

        return numba_get_windowed_version
