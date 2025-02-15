from collections import deque
from pathlib import Path
from typing import Callable, cast
import numba
import numpy as np
import torch

from bcjrformer.codes.galois_field import GaloisField2m
from bcjrformer.utils import get_code_dir
from enum import StrEnum
from numba import njit
import galois


class LinearCodeType(StrEnum):
    POLAR = "POLAR"
    HAMMING = "HAMMING"
    BCH = "BCH"
    CCSDS = "CCSDS"
    LDPC = "LDPC"
    MACKAY = "MACKAY"


class CodeFileExtension(StrEnum):
    ALIST = ".alist"
    TXT = ".txt"


@njit(cache=True)
def min_sum(x, y) -> float:
    if x == np.inf:
        return y
    return np.sign(np.nan_to_num(x * y)) * min(abs(x), abs(y)) + np.log(
        (1 + np.exp(-abs(x + y))) / (1 + np.exp(-abs(x - y)))
    )


_NON_RANDOM_ALIST_SEP = "   "


class LinearCode:

    c_nodes: np.ndarray
    """Check nodes connection matrix"""
    v_nodes: np.ndarray
    """Variable node connection matrix"""
    max_c_degree: int
    """Maximum check node degree"""
    max_v_degree: int
    """Maximum variable node degree"""
    c_degree: np.ndarray
    """Check node degree vector"""
    v_degree: np.ndarray
    """Variable node degree vector"""

    standard_form: bool
    """Whether the code should be transformed to standard form"""

    code_type: LinearCodeType
    """Type of the code"""
    n: int
    """Number of (encoded) bits"""
    k: int
    """Number of information bits"""

    q: int
    """Size of the Galois field"""

    file_path: Path
    """Path to the code file"""
    file_extension: CodeFileExtension
    """Extension of the code file"""

    generator_matrix: np.ndarray
    pc_matrix: np.ndarray
    _tg_distance_matrix: torch.Tensor | None = None
    _numba_log_belief_propagation: Callable[[np.ndarray], np.ndarray] | None = None

    def __init__(
        self,
        code_type: LinearCodeType,
        n: int,
        k: int,
        q: int,
        code_file_path: Path,
        code_file_extension: CodeFileExtension,
        standard_form: bool = False,
        random_weights: bool = False,
    ):
        self.code_type = code_type
        self.n = n
        self.k = k
        self.q = q
        self.m = n - k
        self.file_path = code_file_path
        self.file_extension = code_file_extension
        self.random_weights = random_weights
        self.standard_form = standard_form

        # If we have an alist file we generate TG properties during reading, otherwise we have
        # to derive them from the pc matrix
        # Inits:
        # - pc_matrix
        # - v_nodes, c_nodes
        # - v_degree, c_degree
        # - max_v_degree, max_c_degree
        # - vn_to_cn, cn_to_vn
        if self.file_extension == CodeFileExtension.ALIST:
            _pc_matrix = self.init_from_alist()
        elif self.file_extension == CodeFileExtension.TXT:
            _pc_matrix = np.loadtxt(self.file_path).astype(np.int64)
            self.init_tg_from_pc_matrix(_pc_matrix.view(np.ndarray))
        else:
            raise ValueError(f"Invalid file extension: {self.file_extension}")

        gf = galois.GF(self.q)

        if self.standard_form:
            _pc_matrix = get_systematic_form(_pc_matrix, gf)

            _generator_matrix = np.concatenate(
                [
                    _pc_matrix[:, self.k :].transpose(),
                    np.eye(self.k, dtype=np.int64),
                ],
                1,
                dtype=np.int64,
            )
            self.init_tg_from_pc_matrix(_pc_matrix.view(np.ndarray))
        else:
            # Calculate Generator Matrix from PC Matrix
            _generator_matrix = self.calculate_generator_matrix(_pc_matrix, gf)

        assert (
            gf(_pc_matrix) @ gf(_generator_matrix).T == 0
        ).all(), "Failed to generate valid generator matrix"

        self.generator_matrix = _generator_matrix
        self.pc_matrix = _pc_matrix

    @classmethod
    def from_code_args(
        cls,
        code_type: LinearCodeType,
        n: int,
        k: int,
        q: int = 2,
        code_file_extension: CodeFileExtension | None = None,
        standard_form: bool = False,
        custom_file_name: str | None = None,
        random_weights: bool = False,
    ) -> "LinearCode":

        if code_file_extension is None:
            if code_type in [LinearCodeType.CCSDS, LinearCodeType.MACKAY]:
                code_file_extension = CodeFileExtension.ALIST
            elif code_type in [
                LinearCodeType.POLAR,
                LinearCodeType.BCH,
                LinearCodeType.HAMMING,
                LinearCodeType.LDPC,
            ]:
                code_file_extension = CodeFileExtension.TXT
            else:
                raise ValueError(f"Invalid code type: {code_type}")

        if custom_file_name is not None:
            file_name = custom_file_name
        else:
            file_name = f"{code_type}_N{n}_K{k}" + str(code_file_extension)

        file_path = get_code_dir() / file_name

        if not file_path.exists():
            raise FileNotFoundError(
                f"Code file not found: {file_path}. If the file does not have extension {code_file_extension}, please provide it explicitly."
            )

        return cls(
            code_type,
            n,
            k,
            q,
            file_path,
            code_file_extension,
            standard_form,
            random_weights,
        )

    # n: int, k: int, q: int,
    @classmethod
    def from_vn_to_cn(
        cls, n: int, k: int, q: int, alist_file: Path, vn_to_cn: np.ndarray
    ) -> "LinearCode":
        """Generate a linear code based off vn_to_cn vector

        Args:
            n (int): Number of encoded bits
            k (int): Number of input bits
            q (int): Alphabet size
            vn_to_cn (np.ndarray): An array specifying connections between vn and cn

        Returns:
            LinearCode
        """

        lin_code = cls(
            LinearCodeType.LDPC,
            n,
            k,
            q,
            alist_file,
            CodeFileExtension.ALIST,
            standard_form=False,
            random_weights=True,
        )

        lin_code.set_vn_to_cn(vn_to_cn)

        return lin_code

    def init_tg_from_pc_matrix(self, _pc_matrix: np.ndarray):

        self.v_degree = (_pc_matrix != 0).sum(0)
        self.c_degree = (_pc_matrix != 0).sum(1)

        self.max_c_degree = int(self.c_degree.max().item())
        self.max_v_degree = int(self.v_degree.max().item())

        self.v_nodes = np.full(
            (
                self.n,
                self.max_v_degree,
            ),
            -1,
        )
        self.c_nodes = np.full(
            (
                self.m,
                self.max_c_degree,
            ),
            -1,
        )

        vn_to_cn = np.full((self.n, self.max_v_degree), -1)

        for i in range(self.n):
            v_nodes_i = np.where(_pc_matrix[:, i])[0]

            self.v_nodes[i, : self.v_degree[i]] = v_nodes_i

            if self.random_weights:
                vn_to_cn[i, : self.v_degree[i]] = np.random.randint(
                    1, self.q, self.v_degree[i]
                )
            else:
                vn_to_cn[i, : self.v_degree[i]] = _pc_matrix[v_nodes_i, i]

        for i in range(self.m):
            self.c_nodes[i, : self.c_degree[i]] = np.where(_pc_matrix[i, :])[0]

        self.vn_to_cn = vn_to_cn
        self.cn_to_vn = self.generate_cn_to_vn(vn_to_cn)

    def generate_cn_to_vn(self, vn_to_cn: np.ndarray) -> np.ndarray:
        cn_to_vn = np.full((self.m, self.max_c_degree), -1)

        for j in range(self.n):
            for jj in range(self.v_degree[j]):

                current_check_node = self.v_nodes[j, jj]
                current_vn_to_cn = vn_to_cn[j, jj]

                neighbor_index = (
                    np.asarray(self.c_nodes[current_check_node, :] == j)
                    .nonzero()[0]
                    .item()
                )
                cn_to_vn[current_check_node, neighbor_index] = current_vn_to_cn

        return cn_to_vn

    def _determine_pc_matrix_from_tg(self):

        H = np.zeros((self.m, self.n)).astype(int)

        for v_node in range(self.n):
            for c_node in range(self.v_degree[v_node]):
                c_row = self.v_nodes[v_node, c_node]
                H[c_row, v_node] = self.vn_to_cn[v_node, c_node]

        return H

    # vn to cn can be interpreted as the partiy check values in q-ary form
    def set_vn_to_cn(self, vn_to_cn: np.ndarray):
        if self.q == 2 and not (self.vn_to_cn == 1).all():
            raise ValueError(
                "Variable node to check node messages must be all 1 if q=2"
            )

        self.vn_to_cn = vn_to_cn
        self.cn_to_vn = self.generate_cn_to_vn(vn_to_cn)

        # Determine new Parity Check Matrix from vn_to_cn
        _pc_matrix = self._determine_pc_matrix_from_tg()

        gf = galois.GF(self.q)

        # Derive new generator matrix
        _generator_matrix = self.calculate_generator_matrix(_pc_matrix, gf)

        self.generator_matrix = _generator_matrix
        self.pc_matrix = _pc_matrix
        self.random_weights = False

        # reset numba function but only if q > 2 because otherwise the vn_to_cn cant really change
        if self._numba_log_belief_propagation is not None and self.q > 2:
            self._numba_log_belief_propagation = None

    @property
    def rate(self):
        return self.k / self.n

    def init_from_alist(
        self,
    ):
        """Read a non binary ldpc code from an alist file

        - This differs from the binary / unweighted case, because here we have to read the weights aswell
        """
        with open(self.file_path, "r") as file:
            lines = [line.rstrip() for line in file.readlines()]

        header = np.fromstring(lines[0].rstrip("\n"), dtype=int, sep=" ")

        needs_weights = self.q > 2 and not self.random_weights

        # If q > 2 and we dont set random weights, then we need the weights to be in the alist file
        if needs_weights:
            if len(header) != 3:
                raise ValueError(
                    f"Invalid header in alist file for non-random weights and q > 2: {header}. Expected exactly 3 values."
                )
            elif alist_q := int(header[-1]) != self.q:
                raise ValueError(
                    f"Invalid q value in alist file header for non-random weights and q = {self.q}: {alist_q}"
                )

        if not needs_weights:
            if len(header) == 2:
                col_num, row_num = header
            else:
                col_num, row_num, _ = header
        else:
            col_num, row_num, _ = header

        if not col_num == self.n:
            raise ValueError(
                f"Invalid number of columns in alist file. Expected {self.n}, got {col_num}"
            )

        if not row_num == self.m:
            raise ValueError(
                f"Invalid number of rows in alist file. Expected {self.m}, got {row_num}"
            )

        H = np.zeros((row_num, col_num)).astype(int)

        self.max_v_degree, self.max_c_degree = np.fromstring(
            lines[1], dtype=int, sep=" "
        )

        self.v_nodes = np.full((self.n, self.max_v_degree), -1)
        self.c_nodes = np.full((self.m, self.max_c_degree), -1)

        self.v_degree = np.fromstring(lines[2], dtype=int, sep=" ")
        self.c_degree = np.fromstring(lines[3], dtype=int, sep=" ")

        vn_to_cn = np.full((self.n, self.max_v_degree), -1)

        has_weights = lines[4].count(_NON_RANDOM_ALIST_SEP) > 0

        if not has_weights and needs_weights:
            raise ValueError(
                "Invalid alist file for non-random. Expected weights in the first line, but none found."
            )

        for i in range(self.n):

            if has_weights:
                pos_val_pairs = lines[4 + i].split(_NON_RANDOM_ALIST_SEP)[
                    : self.v_degree[i]
                ]

                if not len(pos_val_pairs) == self.v_degree[i]:
                    raise ValueError(
                        f"Invalid number of values for variable node {i}. Expected {self.v_degree[i]}, got {len(pos_val_pairs)}"
                    )

                pos = np.array([int(pair.split(" ")[0]) for pair in pos_val_pairs]) - 1

                if not self.random_weights:
                    val = np.array([int(pair.split(" ")[1]) for pair in pos_val_pairs])
                else:
                    val = np.random.randint(1, self.q, self.v_degree[i])
            else:
                pos = (
                    np.fromstring(lines[4 + i], dtype=int, sep=" ")[: self.v_degree[i]]
                    - 1
                )

                if len(pos) != self.v_degree[i]:
                    raise ValueError(
                        f"Invalid number of values for variable node {i}. Expected {self.v_degree[i]}, got {len(pos)}"
                    )
                val = np.random.randint(1, self.q, self.v_degree[i])

            self.v_nodes[i, : self.v_degree[i]] = pos
            vn_to_cn[i, : self.v_degree[i]] = val

            H[pos, i] = val

        for i in range(self.m):
            self.c_nodes[i, : self.c_degree[i]] = H[i].nonzero()[0]

        self.cn_to_vn = self.generate_cn_to_vn(vn_to_cn)
        self.vn_to_cn = vn_to_cn

        return H

    @staticmethod
    def graph_to_alist(
        file_path: Path,
        n: int,
        m: int,  # n - k
        q: int,
        max_v_degree: int,
        max_c_degree: int,
        v_degree: np.ndarray,
        c_degree: np.ndarray,
        v_nodes: np.ndarray,
        c_nodes: np.ndarray,
        vn_to_cn: np.ndarray,
        cn_to_vn: np.ndarray,
    ):
        with open(file_path, "w") as f:
            # 1) Header line:
            #    - If q > 2 and we do NOT have random weights, then we write "n m q".
            #    - Otherwise, just write "n m".
            if q > 2:
                f.write(f"{n} {m} {q}\n")
            else:
                f.write(f"{n} {m}\n")

            # 2) Line with max_v_degree, max_c_degree
            f.write(f"{max_v_degree} {max_c_degree}\n")

            # 3) Line with v_degree (space-separated)
            f.write(" ".join(str(x) for x in v_degree) + " \n")

            # 4) Line with c_degree (space-separated)
            f.write(" ".join(str(x) for x in c_degree) + " \n")

            # 5) For each variable node, write either:
            #    - "pos+1 val" pairs (if q > 2), separated by triple-space
            #    - or just "pos+1" (if q = 2), separated by single space
            for i in range(n):
                positions = v_nodes[i, : v_degree[i]]  # Indices of check nodes for VN i

                if q > 2:
                    values = vn_to_cn[i, : v_degree[i]]
                    # Create "pos+1 val" pairs, separated by triple-space
                    line_data = (
                        _NON_RANDOM_ALIST_SEP.join(
                            f"{pos + 1} {val}" for pos, val in zip(positions, values)
                        )
                        + _NON_RANDOM_ALIST_SEP
                    )
                else:
                    # Only output positions (1-based)
                    line_data = " ".join(str(pos + 1) for pos in positions) + " "

                f.write(line_data + "\n")

            for j in range(m):
                positions = c_nodes[j, : c_degree[j]]

                if q > 2:
                    values = cn_to_vn[j, : c_degree[j]]

                    line_data = (
                        _NON_RANDOM_ALIST_SEP.join(
                            f"{pos + 1} {val}" for pos, val in zip(positions, values)
                        )
                        + _NON_RANDOM_ALIST_SEP
                    )

                else:
                    line_data = " ".join(str(pos + 1) for pos in positions) + " "

                f.write(line_data + "\n")

        print(f"Alist file written to: {file_path}")

    def write_alist(self, file_path: Path):
        """
        Write the current parity-check matrix and associated graph structure to an
        .alist file. The format produced matches the expectations of `init_from_alist`.
        """

        self.graph_to_alist(
            file_path,
            self.n,
            self.m,
            self.q,
            self.max_v_degree,
            self.max_c_degree,
            self.v_degree,
            self.c_degree,
            self.v_nodes,
            self.c_nodes,
            self.vn_to_cn,
            self.cn_to_vn,
        )

    def calculate_generator_matrix(
        self,
        pc_matrix: np.ndarray,
        GF,
    ):
        pc_matrix_T = pc_matrix.transpose()
        pc_matrix_I = np.concatenate(
            (pc_matrix_T, np.eye(pc_matrix_T.shape[0], dtype=int)), axis=-1
        )

        gf_pc_matrix_I = GF(pc_matrix_I)

        pc_matrix_I_r, p = row_reduce(gf_pc_matrix_I, GF, ncols=pc_matrix_T.shape[1])

        G = row_reduce(pc_matrix_I_r[p:, pc_matrix_T.shape[1] :], GF)[0]

        return G.view(np.ndarray)

    def calculate_girth(self) -> int | None:
        n = self.n  # Number of variable nodes
        k = self.k  # Dimension of the code
        girth = float("inf")  # Initialize girth to infinity
        total_nodes = 2 * n - k  # Total variable + check nodes

        for start_node in range(total_nodes):
            visited = [False] * total_nodes
            dist = [float("inf")] * total_nodes
            parent = [-1] * total_nodes

            queue = deque()
            visited[start_node] = True
            dist[start_node] = 0
            queue.append(start_node)

            while queue:
                node = queue.popleft()
                dist_node = dist[node]

                # Get neighbors based on node type
                if node < n:
                    # Variable node: connected to check nodes
                    check_indices = torch.nonzero(
                        self.pc_matrix[:, node], as_tuple=False
                    ).squeeze(1)
                    neighbors = (n + check_indices).tolist()
                else:
                    # Check node: connected to variable nodes
                    var_indices = torch.nonzero(
                        self.pc_matrix[node - n, :], as_tuple=False
                    ).squeeze(1)
                    neighbors = var_indices.tolist()

                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        dist[neighbor] = dist_node + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    elif parent[node] != neighbor:
                        # Cycle detected: update girth
                        cycle_length: int = dist_node + dist[neighbor] + 1
                        if cycle_length < girth:
                            girth = cycle_length
                            if girth == 4:  # Early exit for bipartite graphs
                                return girth

        return cast(int, girth) if girth != float("inf") else None

    @staticmethod
    @njit(cache=True)
    def jit_belief_propagation(
        l_a: np.ndarray,
        max_iter: int,
        n: int,
        m: int,
        max_v_degree: int,
        max_c_degree: int,
        c_degree: np.ndarray,
        v_degree: np.ndarray,
        v_nodes: np.ndarray,
        c_nodes: np.ndarray,
    ):
        # reminder: j is max_v_degree, m is max_c_degree

        # messages from variable nodes to check nodes / from check nodes to variable nodes
        l_vc = np.zeros((max_v_degree, n))
        l_cv = np.zeros((max_c_degree, m))

        # extrinsic information
        l_e = np.zeros_like(l_a)

        # temporary message variables
        l_vc_i = np.zeros(max_c_degree)
        l_cv_i = np.zeros(max_v_degree)

        for iter_i in range(max_iter):

            # variable node update
            for i in range(n):
                # get messages from neighbors
                for l in range(v_degree[i]):  # noqa: E741
                    neighbor = v_nodes[l, i]

                    # get message from neighbor
                    for r in range(c_degree[neighbor]):
                        if c_nodes[r, neighbor] == i:
                            l_cv_i[l] = l_cv[r, neighbor]

                # update
                for l in range(v_degree[i]):  # noqa: E741
                    l_vc[l, i] = l_a[i, 0] - l_a[i, 1]  # a priori information
                    for r in range(v_degree[i]):
                        if r != l:
                            l_vc[l, i] += l_cv_i[
                                r
                            ]  # sum of incoming nodes except the current one

            # check node update
            for i in range(m):
                # get messages from neighbors
                for l in range(c_degree[i]):  # noqa: E741
                    neighbor = c_nodes[l, i]

                    # get message from neighbor
                    for r in range(v_degree[neighbor]):
                        if v_nodes[r, neighbor] == i:
                            l_vc_i[l] = l_vc[r, neighbor]

                # update
                for l in range(c_degree[i]):  # noqa: E741
                    l_cv[l, i] = np.inf
                    for r in range(c_degree[i]):
                        if r != l:
                            if np.isnan(min_sum(l_cv[l, i], l_vc_i[r])):
                                assert False, "NaN value encountered in min_sum"
                            l_cv[l, i] = min_sum(l_cv[l, i], l_vc_i[r])

            # compute final APP estimate
            for i in range(n):
                for l in range(v_degree[i]):  # noqa: E741
                    neighbor = v_nodes[l, i]

                    for r in range(c_degree[neighbor]):
                        if c_nodes[r, neighbor] == i:
                            l_e[i, 0] += l_cv[r, neighbor]

            # check if the estimated word is a codeword

            is_codeword = True
            for i in range(m):
                par = 0
                for l in range(c_degree[i]):  # noqa: E741
                    if (l_e[c_nodes[l, i], 1] + l_a[c_nodes[l, i], 1]) > (
                        l_e[c_nodes[l, i], 0] + l_a[c_nodes[l, i], 0]
                    ):
                        par += 1
                if par % 2 == 1:
                    is_codeword = False
                    break

            if is_codeword:
                return l_e

        return l_e

    @staticmethod
    @njit(cache=True)
    def log_normalize(l_a_row):
        # acc = np.logaddexp.reduce(l_a_row)
        acc = -np.inf
        for i in range(len(l_a_row)):
            acc = np.logaddexp(acc, l_a_row[i])

        # NOTE: Check this with Franziska
        # If all values are -inf, we return a uniform distribution
        # However: We should afaik never get to this point in the bcjr algorithm
        if acc == -np.inf:
            return np.full_like(l_a_row, np.log(1 / len(l_a_row)))

        return l_a_row - acc

    def _compile_numba_disc_conv_log(
        self, gf: GaloisField2m
    ) -> numba.types.FunctionType:
        q = self.q
        gf_add = gf.add

        def disc_conv_log(x, y) -> np.ndarray:
            z = np.full_like(x, -np.inf)

            for u in range(q):
                for v in range(q):
                    ix = gf_add(u, v)
                    sum_xy = x[u] + y[v]
                    z[ix] = np.logaddexp(z[ix], sum_xy)
            return z

        return njit(disc_conv_log)

    def _compile_numba_log_belief_propagation(
        self, max_iter, gf: GaloisField2m
    ) -> Callable[[np.ndarray], np.ndarray]:
        _disc_conv_log = self._compile_numba_disc_conv_log(gf)

        _n = self.n
        _m = self.m
        _max_v_degree = self.max_v_degree
        _max_c_degree = self.max_c_degree
        _c_degree = self.c_degree
        _v_degree = self.v_degree
        _v_nodes = self.v_nodes
        _c_nodes = self.c_nodes
        _cn_to_vn = self.cn_to_vn
        _vn_to_cn = self.vn_to_cn
        _q = self.q
        _log_normalize = self.log_normalize

        # define gf functions to make them class-independent
        gf_div = gf.div
        gf_mult = gf.mult
        gf_add = gf.add

        def log_belief_propagation(l_a: np.ndarray):
            # reminder: j is max_v_degree, m is max_c_degree

            # messages from variable nodes to check nodes / from check nodes to variable nodes
            m_vc = np.zeros((_n, _max_v_degree, _q))
            m_cv = np.zeros((_m, _max_c_degree, _q))

            m_cv_tmp = np.zeros((_max_c_degree, _q))
            m_vc_tmp = np.zeros((_max_v_degree, _q))

            curr_msg = np.zeros(_q)

            # fill input in messages
            for j in range(_n):
                l_a[j] = _log_normalize(l_a[j])

                # NOTE: Not sure this works (PS: I think it does)
                m_vc[j, :, :] = l_a[j]

            chat = np.zeros(_n, dtype=np.uint8)

            q_arr = np.arange(_q)

            for it in range(max_iter):

                # check node update
                for i in range(_m):
                    # get neighbor's messages
                    for ii in range(_c_degree[i]):

                        curr_vn = _c_nodes[i, ii]
                        curr_hij = _cn_to_vn[i, ii]

                        assert (
                            curr_vn != -1
                        ), "Invalid check node to variable node message"

                        # NOTE: Technically np.where is not the best way, since we only have one value
                        # => Numba function could be faster
                        neighbor_index = (
                            np.asarray(_v_nodes[curr_vn] == i).nonzero()[0].item()
                        )

                        gf_perm = gf_div(q_arr, curr_hij)

                        m_cv_tmp[ii, :] = m_vc[curr_vn, neighbor_index, gf_perm]

                    # update
                    for ii in range(_c_degree[i]):
                        start = 2 if ii == 0 else 1
                        curr_msg = m_cv_tmp[start - 1]

                        for iii in range(start, _c_degree[i]):
                            if iii != ii:
                                curr_msg = _disc_conv_log(curr_msg, m_cv_tmp[iii])

                        m_cv[i, ii] = curr_msg

                # variable node update
                for j in range(_n):
                    # get neighbor's messages
                    for jj in range(_v_degree[j]):
                        curr_cn = _v_nodes[j, jj]
                        curr_hij = _vn_to_cn[j, jj]

                        assert (
                            curr_cn != -1
                        ), "Invalid variable node to check node message"

                        # NOTE: Technically np.where is not the best way, since we only have one value
                        # => Numba function could be faster
                        neighbor_index = (
                            np.asarray(_c_nodes[curr_cn] == j).nonzero()[0].item()
                        )

                        gf_perm = gf_mult(q_arr, curr_hij)

                        m_vc_tmp[jj, :] = m_cv[curr_cn, neighbor_index, gf_perm]

                    # update
                    for jj in range(_v_degree[j]):
                        curr_msg = l_a[j].copy()

                        for jjj in range(_v_degree[j]):
                            if jjj != jj:
                                curr_msg += m_vc_tmp[jjj]

                        m_vc[j, jj] = _log_normalize(curr_msg)

                l_eo = l_a.copy()

                for j in range(_n):
                    for jj in range(_v_degree[j]):
                        curr_cn = _v_nodes[j, jj]
                        curr_hij = _vn_to_cn[j, jj]

                        for ii in range(_c_degree[curr_cn]):
                            if _c_nodes[curr_cn, ii] == j:
                                gf_perm = gf_mult(q_arr, curr_hij)
                                l_eo[j] += m_cv[curr_cn, ii, gf_perm]

                    # here we basically get logits which we need to normalize to get probabilities?
                    # => We take the maximum value as the most likely codeword
                    l_eo[j] = _log_normalize(l_eo[j])
                    chat[j] = l_eo[j].argmax()

                is_codeword = True
                for i in range(_m):
                    parity = 0
                    for ii in range(_c_degree[i]):
                        curr_hij = _cn_to_vn[i, ii]
                        curr_vn = _c_nodes[i, ii]
                        parity = gf_add(
                            parity,
                            gf_mult(
                                curr_hij,
                                chat[curr_vn],
                            ),
                        )
                    if parity != 0:
                        is_codeword = False
                        break

                if is_codeword:
                    return l_eo

            return l_eo

        # TODO: Debug
        # return log_belief_propagation
        return njit(log_belief_propagation)

    def belief_propagation(
        self, l_a: np.ndarray, max_iter: int, gf: GaloisField2m
    ) -> np.ndarray:
        if self._numba_log_belief_propagation is None:
            self._numba_log_belief_propagation = (
                self._compile_numba_log_belief_propagation(max_iter, gf)
            )

        return self._numba_log_belief_propagation(l_a.copy())


#############################################
def row_reduce(mat, GF, ncols=None):
    assert mat.ndim == 2
    ncols = mat.shape[1] if ncols is None else ncols
    mat_row_reduced = mat.copy()
    p = 0
    for j in range(ncols):
        # Find the first non-zero entry in column j starting at row p
        rows = np.nonzero(mat_row_reduced[p:, j])[0]
        if rows.size == 0:
            continue  # No pivot in this row

        pivot_row = p + rows[0]

        if pivot_row != p:
            mat_row_reduced[[p, pivot_row]] = mat_row_reduced[[pivot_row, p]]

        # Normalize pivot element to make leading entry 1
        pivot_element = mat_row_reduced[p, j]
        if pivot_element == 0:
            continue  # Skip if pivot is zero (shouldn't happen)

        inv_pivot = GF(1) / pivot_element  # Compute multiplicative inverse
        mat_row_reduced[p] = mat_row_reduced[p] * inv_pivot

        # Eliminate all other entries in column j
        for i in range(mat_row_reduced.shape[0]):
            if i != p and mat_row_reduced[i, j] != 0:
                factor = mat_row_reduced[i, j]
                mat_row_reduced[i] -= factor * mat_row_reduced[p]
        p += 1
        if p == mat_row_reduced.shape[0]:
            break
    return mat_row_reduced, p


def get_systematic_form(mat: np.ndarray, GF):
    mat_row_reduced, rank = row_reduce(GF(mat), GF)

    # Rearrange columns to enforce identity submatrix
    n, m = mat_row_reduced.shape
    pivots = []
    for i in range(rank):
        for j in range(m):
            if mat_row_reduced[i, j] != 0 and j not in pivots:
                pivots.append(j)
                break

    # Reorder columns to place pivots first
    non_pivots = [j for j in range(m) if j not in pivots]
    new_order = pivots + non_pivots
    systematic_mat = mat_row_reduced[:, new_order]

    return systematic_mat


def main():
    code_file_path = get_code_dir().iterdir()
    for code_file in code_file_path:
        if code_file.suffix != ".txt" and code_file.suffix != ".alist":
            continue

        code_file_name = code_file.name

        code_n = int(code_file_name.split("_")[1][1:])
        code_k = int(code_file_name.split("_")[-1][1:].split(".")[0])

        try:
            code_code_type = LinearCodeType(code_file_name.split("_")[0])
        except ValueError as e:
            raise ValueError(
                f"Code does not have a valid code type: {code_file_name}"
            ) from e

        code = LinearCode(
            code_code_type,
            code_n,
            code_k,
            2,
            code_file_path=code_file,
            code_file_extension=CodeFileExtension(code_file.suffix),
        )

        print(code.code_type, code.n, code.k)
        print(code.pc_matrix, code.generator_matrix)


if __name__ == "__main__":
    main()
