from typing import Callable, Protocol
import numpy as np


import random


class Channel(Protocol):
    def transmit(self, m: list[int]) -> tuple[list[int], list[str]]: ...


class IDSChannel:

    _numba_transmit: (
        Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None
    ) = None

    def __init__(
        self,
        q: int,
        p_i: float,
        p_d: float,
        p_s: float,
    ):
        self.q = q
        self.p_i = p_i
        self.p_d = p_d
        self.p_s = p_s
        self.p_t = 1 - p_i - p_d

    def numba_compile_transmit(self):
        p_i = self.p_i
        p_d = self.p_d
        p_s = self.p_s
        q = self.q

        def _transmit(m: np.ndarray, M: int) -> tuple[np.ndarray, np.ndarray]:
            """Transmit a message through the channel

            Args:
                m (np.ndarray): message to transmit
                M (int): number of times to transmit

            Returns:
                tuple[np.ndarray, np.ndarray]: Transmitted messages and and their lengths
            """
            out: list[list[int]] = []
            for _ in range(M):
                out_i = []
                i = 0
                while i < len(m):
                    p_transmission = random.random()
                    # insertions
                    if p_transmission < p_i:
                        out_i.append(random.randint(0, q - 1))
                        i -= 1
                    # deletions
                    elif p_transmission < p_i + p_d:
                        pass

                    # transmissions
                    else:
                        p_substitution = random.random()
                        # substitutions
                        if p_substitution < p_s:
                            s_symbol = (m[i] + random.randint(1, q - 1)) % q
                            out_i.append(s_symbol)
                        # no substitutions
                        else:
                            out_i.append(m[i])
                    i += 1
                out.append(out_i)
            N = np.array([len(o) for o in out], dtype=np.int16)
            max_length = N.max()

            out_np = np.full((M, max_length), fill_value=-1, dtype=np.int8)
            for i, o in enumerate(out):
                out_np[i, : N[i]] = o

            return out_np, N

        return _transmit

    def numba_transmit(self, m: np.ndarray, M: int) -> tuple[np.ndarray, np.ndarray]:
        """Transmit a message through the channel

        Args:
            m (np.ndarray): message to transmit
            A (int): number of times to transmit

        Returns:
            np.ndarray: Transmitted messages
        """
        if self._numba_transmit is None:
            self._numba_transmit = self.numba_compile_transmit()

        return self._numba_transmit(m, M)

    def transmit(self, m: list[int] | np.ndarray) -> tuple[list[int], list[str]]:
        events: list[str] = []
        out = []

        i = 0
        while i < len(m):
            p = random.random()
            if p < self.p_i:
                random_symbol = random.randint(0, self.q - 1)
                out.append(random_symbol)
                events.append(f"i{random_symbol}")
                i -= 1
            elif self.p_i <= p < self.p_i + self.p_d:
                events.append(f"d{m[i]}")
            else:
                p_s = random.random()
                if p_s < self.p_s:
                    s_symbol = (m[i] + random.randint(1, self.q - 1)) % self.q

                    assert (
                        s_symbol != m[i]
                    ), f"Substitution with identical symbol {s_symbol} == {m[i]}"

                    out.append(s_symbol)
                    events.append(f"s{s_symbol}")
                else:
                    events.append("t")
                    out.append(m[i])
            i += 1
        return out, events

    def int_to_vector(self, i: int, v: np.ndarray, n: int):
        for j in range(n):
            v[j] = i % self.q
            i //= self.q

    def vector_to_int(self, v: np.ndarray, n: int) -> int:
        return np.dot(v[::-1], np.power(self.q, np.arange(n)))

    def prob_yx(self, x: np.ndarray, y: np.ndarray, F: np.ndarray, Ft: np.ndarray):
        n = len(x)
        m = len(y)

        Ft[0][0] = 1

        for r in range(m + 1):
            for p in range(n + 1):
                # deletions
                if p > 0:
                    Ft[p][r] += self.p_d * Ft[p - 1][r]
                # substitutions
                if p > 0 and r > 0:
                    Ft[p][r] += Ft[p - 1][r - 1] * (
                        (self.p_t * (1 - self.p_s))
                        if x[p - 1] == y[r - 1]
                        else (self.p_t * self.p_s) / (self.q - 1)
                    )
                F[p][r] = Ft[p][r]

                # insertions
                if r > 0:
                    Ft[p][r] += self.p_i * Ft[p][r - 1] / self.q

    def get_final_drift_statistics(
        self,
    ) -> tuple[float, float]:
        """Calculate avg and var of the final drift distribution"""
        avg = (self.p_i - self.p_d) / (1 - self.p_i)

        var = (1 - self.p_d) * (self.p_d + self.p_i) / ((1 - self.p_i) ** 2)

        return avg, var


class FixedDeletionChannel:
    def __init__(
        self,
        q: int,
        p_s: float,
        del_count: int,
    ):
        self.q = q
        self.del_count = del_count
        self.p_s = p_s

    def transmit(self, m: list[int]) -> tuple[list[int], list[str]]:
        events: list[str] = [""] * len(m)

        np_m = np.array(m)

        kept_indices = sorted(random.sample(range(len(m)), len(m) - self.del_count))

        out = np_m[kept_indices]

        i = 0
        j = 0
        while i < len(m):
            if i in kept_indices:
                p_s = random.random()
                if p_s < self.p_s:
                    out[j] = (out[j] + random.randint(1, self.q - 1)) % self.q
                    events[i] = "s"
                else:
                    events[i] = "t"
                j += 1
            else:
                events[i] = f"d{m[i]}"
            i += 1

        return out.tolist(), events


if __name__ == "__main__":
    ids_channel = IDSChannel(2, 0.1, 0.1, 0.1)

    message = np.random.randint(0, 2, 10)
    print(message)

    out, N = ids_channel.numba_transmit(message, 10)
    print(out, N)
