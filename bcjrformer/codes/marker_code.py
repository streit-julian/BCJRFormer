from typing import Callable
import numpy as np
import torch
import numba as nb


class MarkerCode:
    """Class to interact with Marker Codes"""

    _numba_encode: None | Callable[[np.ndarray], np.ndarray] = None

    def __init__(self, marker: list[int], N_c: int, p: int, T: int):
        """Constructor for MarkerCode

        Args:
            marker (list[int]): The marker code, e. g. [0, 0, 1], [0, 3, 0], ...
            N_c (int): The number of bits between two markers
            p (int): Should be the the number of bits to encode the alphabet
            (e. g. 2 bits if q = 4, 3 bits if q = 8, ...)
            T (int): The number of time steps to encode the message
            (i. e. the number of bits in the input message)
        """

        if N_c > T:
            raise ValueError("N_c must be smaller than T")

        self.marker = marker

        if any([m >= 1 << p for m in marker]):
            raise ValueError(
                f"Marker code must be in the range of 0 to {2**p - 1} for p = {p}"
            )

        self.p = p

        self.q = 1 << p

        self.T = T

        self.N_c = N_c

        self.marker_length = len(marker)

        self.block_length = N_c + self.marker_length

        self.encoded_length = T + (T // N_c * self.marker_length)

        self.log_likelihoods = self._get_ll()

    def get_prior_information(self) -> np.ndarray:
        return self.log_likelihoods

    def _get_ll(self) -> np.ndarray:
        """Calculates the log likelihoods for the marker code

        Returns:
            np.ndarray: The log likelihoods
        """

        ll = np.full((self.encoded_length, self.q), -np.inf)

        # set the log likelihoods of the marker code to 0
        for t in range(self.encoded_length // self.block_length):
            # insert the (input/even) log likelihoods
            ll[t * self.block_length : t * self.block_length + self.N_c] = np.log(
                1 / self.q
            )

            ll[t * self.block_length + self.N_c : (t + 1) * self.block_length][
                np.arange(len(self.marker)), self.marker
            ] = 0
        else:
            ll[(t + 1) * self.block_length :] = np.log(1 / self.q)

        return ll

    def get_marker(self, index: int) -> tuple[bool, int]:
        """Returns if the bit at index is a marker bit and the marker index

        Args:
            index (int): The index to check

        Returns:
            tuple[bool, int]: If the bit is a marker bit and the marker index
        """
        if index % self.block_length < self.N_c:
            return False, -1
        else:
            return True, self.marker[(index % self.block_length) - self.N_c]

    def get_marker_mask(self) -> np.ndarray:
        """Returns a mask for the marker bits

        Returns:
            np.ndarray: The marker mask
        """
        mask = np.zeros(self.encoded_length, dtype=bool)

        for t in range(self.encoded_length // self.block_length):
            mask[t * self.block_length + self.N_c : (t + 1) * self.block_length] = True

        return mask

    def _numba_encode_func(self) -> Callable[[np.ndarray], np.ndarray]:
        """Returns the numba version of the encode function"""

        block_length = self.block_length
        N_c = self.N_c
        marker = np.array(self.marker, dtype=np.int64)
        encoded_length = self.encoded_length

        @nb.njit
        def numba_encode(message: np.ndarray) -> np.ndarray:
            enc = np.zeros((message.shape[0], encoded_length), dtype=np.int64)

            for t in range(encoded_length // block_length):
                # insert the message
                enc[:, t * block_length : t * block_length + N_c] = message[
                    :, t * N_c : (t + 1) * N_c
                ]

                # insert the marker code
                enc[:, t * block_length + N_c : (t + 1) * block_length] = marker
            # if there is still some message left, i. e. the message length is not a multiple of N_c
            # we insert the remaining message
            else:
                enc[:, (t + 1) * block_length :] = message[:, (t + 1) * N_c :]

            return enc

        return numba_encode

    def numba_encode(self, message: np.ndarray) -> np.ndarray:
        if self._numba_encode is None:
            self._numba_encode = self._numba_encode_func()

        return self._numba_encode(message)

    def encode(self, message: np.ndarray) -> np.ndarray:
        """Encodes a message using the Marker Code

        - Inserts the marker code between the blocks of the message

        Args:
            message (np.ndarray): The msg to encode.

        Returns:
            np.ndarray: The encoded message
        """
        enc = np.zeros(self.encoded_length, dtype=int)

        for t in range(self.encoded_length // self.block_length):
            # insert the message
            enc[t * self.block_length : t * self.block_length + self.N_c] = message[
                t * self.N_c : (t + 1) * self.N_c
            ]

            # insert the marker code
            enc[t * self.block_length + self.N_c : (t + 1) * self.block_length] = (
                self.marker
            )
        # if there is still some message left, i. e. the message length is not a multiple of N_c
        # we insert the remaining message
        else:
            enc[(t + 1) * self.block_length :] = message[(t + 1) * self.N_c :]

        return enc

    def decode(self, message: np.ndarray) -> np.ndarray:
        """Decodes a message using the Marker Code

        - Removes the marker code between the blocks of the message

        Args:
            message (np.ndarray): A message with marker code

        Returns:
            np.ndarray: The decoded message
        """
        dec = np.zeros((self.T, message.shape[1]), dtype=message.dtype)

        for t in range(self.encoded_length // self.block_length):
            dec[t * self.N_c : (t + 1) * self.N_c] = message[
                t * self.block_length : t * self.block_length + self.N_c
            ]
        else:
            dec[(t + 1) * self.N_c :] = message[(t + 1) * self.block_length :]

        return dec

    def decode_model(self, x_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Torch version of decode (intended for batch processing)"""

        dec = torch.zeros(
            (x_pred.shape[0], self.T), dtype=x_pred.dtype, device=x_pred.device
        )
        markers = torch.zeros(
            (
                x_pred.shape[0],
                (self.encoded_length // self.block_length) * self.marker_length,
            ),
            dtype=x_pred.dtype,
            device=x_pred.device,
        )

        for t in range(self.encoded_length // self.block_length):
            dec[:, t * self.N_c : (t + 1) * self.N_c] = x_pred[
                :, t * self.block_length : t * self.block_length + self.N_c
            ]
            markers[:, t * self.marker_length : (t + 1) * self.marker_length] = x_pred[
                :,
                t * self.block_length + self.N_c : (t + 1) * self.block_length,
            ]
        else:
            dec[:, (t + 1) * self.N_c :] = x_pred[:, (t + 1) * self.block_length :]

        return dec, markers

    def get_marker_sequence_torch(self) -> torch.Tensor:
        """Returns all marker sequences for the marker code concatenated"""

        marker = torch.tensor(self.marker, dtype=torch.float64)

        return marker.repeat(self.encoded_length // self.block_length)


if __name__ == "__main__":
    from bcjrformer.codes.linear_code import LinearCode
    from bcjrformer.codes.linear_code import LinearCodeType

    linear_code = LinearCode.from_code_args(
        LinearCodeType.HAMMING,
        n=7,
        k=4,
        q=2,
    )

    marker_code = MarkerCode(marker=[0, 0], N_c=2, p=1, T=8)

    print(marker_code.encode(np.array([1, 1, 1, 1, 1, 1, 1, 1])))
