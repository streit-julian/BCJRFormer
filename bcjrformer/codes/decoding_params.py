import numpy as np
from numpy.typing import NDArray


from dataclasses import dataclass

from bcjrformer.channels import IDSChannel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bcjrformer.codes.convolutional_code import (
        ConvolutionalCode,
    )


@dataclass
class DecodingParameters:

    A: int
    "number of received strands"

    N: np.ndarray
    "length of each strand / received codeword"

    max_Y_length: int
    "maximum length of the output Block"

    max_N: int
    "maximum received codeword length"

    d_min: NDArray[np.int_]
    "minimum drift at time t"

    d_max: NDArray[np.int_]
    "maximum drift at time t"

    max_insertions: int
    "maximum number of insertions per symbol"

    num_drift_states: NDArray[np.int_]
    "number of drift states at time t"

    num_relative_drift_states: int
    "number of relative drift states"

    maximum_drift_range: int
    "maximum drift range"


def get_sequence_drift(
    Ni: int,
    encoded_length: int,
    n_trellis: int,
    cum_N: np.ndarray,
    avg_drift_per_symbol: float,
    maximum_drift_range: float,
    max_insertions: int,
) -> np.ndarray:

    d = np.zeros((2, n_trellis + 1), dtype=int)

    t = np.arange(n_trellis + 1)

    d[0] = np.maximum(
        cum_N * avg_drift_per_symbol - maximum_drift_range,
        np.maximum(-cum_N, Ni - encoded_length - (n_trellis - t) * max_insertions),
    ).astype(int)

    d[1] = np.minimum(
        cum_N * avg_drift_per_symbol + maximum_drift_range,
        np.minimum(cum_N * max_insertions, Ni - cum_N),
    ).astype(int)

    return d


def create_decoding_params(
    conv_code: "ConvolutionalCode",
    r: np.ndarray,
    N: np.ndarray,
    ids_channel: IDSChannel,
    max_insertions: int,
    std_mult: float = 3.5,
) -> tuple[DecodingParameters, np.ndarray]:
    """Create the decoding parameters

    - Also cuts of the received codeword at the right length

    Args:
        conv_code (ConvolutionalCode): The Convolutional Code
        r (np.ndarray): The received codeword
        ids_channel (IDSChannel): The IDS Channel
        max_insertions (int): The maximum number of insertions per symbol

    Returns:
        tuple[DecodingParameters, np.ndarray]:
            - The decoding parameters
            - The received codeword ( Possibly truncated / extended )
    """

    A = len(r)
    max_Y_length = conv_code.n_v * (1 + max_insertions)

    avg, var = ids_channel.get_final_drift_statistics()

    # NOTE: if p_i = p_d then this resolves to the formula on page 12 of the paper
    maximum_drift_range = std_mult * np.sqrt(var * conv_code.encoded_length)

    d_min = np.zeros((A, conv_code.B + 1), dtype=int)
    d_max = np.zeros((A, conv_code.B + 1), dtype=int)

    # Truncate / Extend codeword to a viable length
    num_drift_states = np.ones(conv_code.B + 1, dtype=int)
    for i, ri in enumerate(r):

        max_length = int(conv_code.encoded_length * (1 + avg) + maximum_drift_range)

        min_length = int(conv_code.encoded_length * (1 + avg) - maximum_drift_range)

        if len(ri) > max_length:
            ri[max_length:] = -1
            N[i] = max_length
        if len(ri) < min_length:
            ri[N[i] : min_length] = 0
            N[i] = min_length
        d = get_sequence_drift(
            N[i],
            conv_code.encoded_length,
            conv_code.B,
            conv_code.cum_N,
            avg,
            maximum_drift_range,
            max_insertions,
        )

        d_min[i] = d[0]
        d_max[i] = d[1]

        num_drift_states *= d[1] - d[0] + 1

    max_N = N.max()

    # the number of relative drift states
    # is the number of outgoing drift edges per state
    num_relative_drift_states = np.pow(conv_code.n_v * (1 + max_insertions) + 1, A)

    return (
        DecodingParameters(
            A=A,
            N=N,
            max_Y_length=max_Y_length,
            max_N=max_N,
            d_min=d_min,
            d_max=d_max,
            max_insertions=max_insertions,
            num_drift_states=num_drift_states,
            num_relative_drift_states=num_relative_drift_states,
            maximum_drift_range=maximum_drift_range,
        ),
        r,
    )


def truncate_or_extend_codeword(
    r: np.ndarray, N: np.ndarray, encoded_length, avg: float, maximum_drift_range
) -> tuple[np.ndarray, np.ndarray]:

    max_length = int(encoded_length * (1 + avg) + maximum_drift_range)

    min_length = int(encoded_length * (1 + avg) - maximum_drift_range)
    for i, ri in enumerate(r):
        if len(ri) > max_length:
            ri[max_length:] = -1
            N[i] = max_length
        if len(ri) < min_length:
            ri[N[i] : min_length] = 0
            N[i] = min_length

    return r, N
