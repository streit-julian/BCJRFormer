from typing import TypedDict
import numba
import numpy as np


class LookupTable(TypedDict):
    EXP: np.ndarray
    LOG: np.ndarray


class GFLookupTables:

    _LOOKUP_TABLES: dict[int, LookupTable] = {
        2: {
            "EXP": np.array([1, 2, 3, 1, 2, 3, 1, 0]),
            "LOG": np.array([0, 0, 1, 2]),
        },
        3: {
            "EXP": np.array([1, 2, 4, 3, 6, 7, 5, 1, 2, 4, 3, 6, 7, 5, 1, 0]),
            "LOG": np.array([0, 0, 1, 3, 2, 6, 4, 5]),
        },
        4: {
            "EXP": np.array(
                [
                    1,
                    2,
                    4,
                    8,
                    3,
                    6,
                    12,
                    11,
                    5,
                    10,
                    7,
                    14,
                    15,
                    13,
                    9,
                    1,
                    2,
                    4,
                    8,
                    3,
                    6,
                    12,
                    11,
                    5,
                    10,
                    7,
                    14,
                    15,
                    13,
                    9,
                    1,
                    0,
                ]
            ),
            "LOG": np.array([0, 0, 1, 4, 2, 8, 5, 10, 3, 14, 9, 7, 6, 13, 11, 12]),
        },
    }

    def __init__(self, m: int):
        self.m = m
        self.EXP = self._LOOKUP_TABLES[m]["EXP"]
        self.LOG = self._LOOKUP_TABLES[m]["LOG"]


class GaloisField2m:
    """Galois Field Implementation supporting Galois Fields up to 2**4 with numba"""

    def __init__(self, q: int):
        assert np.log2(q) % 1 == 0, "Only field 2^n are supported for q"

        self.q = q
        self.m = int(np.log2(q))

        # generate numba compileable static methods
        self.add = np.bitwise_xor
        self.subtract = np.bitwise_xor

        if self.m > 4:
            raise NotImplementedError("Only fields up to 2**4 are supported")

        if self.m > 1:
            lookup_tables = GFLookupTables(self.m)

            # we use the existing galois field implementation's lookup tables to simplify things a bit
            # self.gf = galois.GF(q, compile="jit-lookup")

            self._EXP = lookup_tables.EXP
            self._LOG = lookup_tables.LOG

            self.mult = self._mult()
            self.div = self._div()
        else:
            self.mult = np.bitwise_and
            self.reciprocal = self._reciprocal_bin()

            # we need to implement division ourselves to cover the case of 0 as divisor
            self.div = self._div_bin()

    def _reciprocal_bin(self) -> numba.types.FunctionType:
        def reciprocal(a: int) -> int:
            if a == 0:
                raise ZeroDivisionError(
                    "Cannot compute the multiplicative inverse of 0 in a Galois field."
                )
            return 1

        return numba.vectorize(["int64(int64)"], nopython=True)(reciprocal)

    def _div_bin(self) -> numba.types.FunctionType:
        def div(a: int, b: int) -> int:
            if b == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return a & b

        return numba.vectorize(["int64(int64, int64)"], nopython=True)(div)

    def _div(self) -> numba.types.FunctionType:
        _EXP = self._EXP
        _LOG = self._LOG
        _ORDER = self.q

        def div(a: int, b: int) -> int:
            if a == 0:
                return 0
            if b == 0:
                raise ZeroDivisionError("Cannot divide by zero")

            m = _LOG[a]
            n = _LOG[b]

            return _EXP[(_ORDER - 1) + m - n]

        return numba.vectorize(["int64(int64, int64)"], nopython=True)(div)

    def _mult(self) -> numba.types.FunctionType:
        _EXP = self._EXP
        _LOG = self._LOG

        def mult(a: int, b: int) -> int:
            if a == 0 or b == 0:
                return 0
            m = _LOG[a]
            n = _LOG[b]

            return _EXP[m + n]

        return numba.vectorize(["int64(int64, int64)"], nopython=True)(mult)
