from enum import StrEnum
from functools import partial
import json
import logging
import os
from pathlib import Path

import typer

from bcjrformer.codes.convolutional_code import (
    MarkerConvolutionalCode,
)
from bcjrformer.codes.decoding_params import (
    create_decoding_params,
)
from bcjrformer.codes.linear_code import CodeFileExtension, LinearCodeType, LinearCode
import numpy as np
from bcjrformer.channels import IDSChannel

from bcjrformer.codes.galois_field import GaloisField2m
from bcjrformer.cli.app import app
from bcjrformer.codes.marker_code import MarkerCode
import multiprocessing as mp
from typing import Callable
from datetime import datetime


class ConvDecoderType(StrEnum):
    BCJR = "bcjr"
    VITERBI = "viterbi"


class DecodingMethod(StrEnum):
    JOINT = "joint"
    SEPARATE = "separate"


def single_transmission_marker(
    r: np.ndarray,
    N: np.ndarray,
    ldpc_offset: np.ndarray,
    joint_decoding: bool,
    ldpc_code: LinearCode,
    marker_code: MarkerCode,
    conv_code: MarkerConvolutionalCode,
    ids_channel: IDSChannel,
    ldpc_max_iterations: int,
    max_insertions: int,
    ldpc_gf: GaloisField2m,
) -> tuple[np.ndarray, int, np.ndarray, int]:
    # Set up extrinsic information:

    # A priori input bit information
    L_ai = marker_code.log_likelihoods.copy()

    # extrinsic inner code information
    L_ei = np.zeros((conv_code.B, ldpc_code.q))

    # A priori outer code information
    L_ao = np.zeros((ldpc_code.n, ldpc_code.q))

    if not joint_decoding:
        for i, ri in enumerate(r):
            ri_unsqueezed = ri[None, :]
            Ni_unsqueezed = N[None, i]

            dp, l_ri = create_decoding_params(
                conv_code, ri_unsqueezed, Ni_unsqueezed, ids_channel, max_insertions
            )

            L_ei_tmp = conv_code.full_numba_bcjr_ids(l_ri, L_ai, dp, ids_channel)

            L_ei += L_ei_tmp - L_ai
            L_ei[np.isneginf(L_ai)] = -np.inf
    else:
        dp, r = create_decoding_params(conv_code, r, N, ids_channel, max_insertions)
        L_ei_tmp = conv_code.full_numba_bcjr_ids(r, L_ai, dp, ids_channel)

        with np.errstate(invalid="ignore"):
            L_ei = L_ei_tmp - L_ai
        L_ei[np.isneginf(L_ai)] = -np.inf

    # remove marker information
    L_ei = marker_code.decode(L_ei_tmp)

    # permute using the given ldpc offset (GF(q) version)
    gf_permutation = ldpc_gf.add(
        np.tile(np.arange(ldpc_code.q), (ldpc_code.n, 1)), ldpc_offset[:, np.newaxis]
    )

    # outer code information (remove ldpc offset which is used to seperate code blocks more easily)
    L_ao = L_ei[np.arange(ldpc_code.n)[:, None], gf_permutation]

    bcjr_dec = L_ao.argmax(axis=1)

    L_eo = ldpc_code.belief_propagation(L_ao, ldpc_max_iterations, ldpc_gf)

    # NOTE: This is unnecessary, we can just do the same thing as for the BCJR
    L_eo[np.arange(ldpc_code.n)[:, None], gf_permutation] = L_eo

    dec = np.argmax(L_eo, axis=1)

    # The difference between the decoded message and the original message is the number of errors
    return (
        dec,
        (dec != ldpc_offset).sum(),
        bcjr_dec,
        (bcjr_dec != np.zeros_like(bcjr_dec)).sum(),
    )


def run_simulation_single_process(
    A: int,
    iterations: int,
    joint_decoding: bool,
    ldpc_bp_iterations: int,
    max_insertions: int,
    ids_channel: IDSChannel,
    ldpc_code: LinearCode,
    ldpc_gf: GaloisField2m,
    marker_code: MarkerCode,
    conv_code: MarkerConvolutionalCode,
):
    num_iterations = num_errors = num_bit_errors = num_bcjr_errors = (
        num_bcjr_bit_errors
    ) = 0

    for i in range(iterations):
        # conv_code.new_random_offset()
        ldpc_offset = np.random.randint(0, ldpc_code.q, ldpc_code.n)

        marker_enc = marker_code.encode(ldpc_offset)

        enc = conv_code.encode(marker_enc)

        r, N = ids_channel.numba_transmit(enc, A)

        dec, errs, bcjr_dec, bcjr_errs = single_transmission_marker(
            r,
            N,
            ldpc_offset,
            joint_decoding,
            ldpc_code,
            marker_code,
            conv_code,
            ids_channel,
            ldpc_bp_iterations,
            max_insertions,
            ldpc_gf,
        )

        num_iterations += 1  # redundant but helpful for bookkeeping
        num_errors += errs > 0
        num_bit_errors += errs

        num_bcjr_errors += bcjr_errs > 0
        num_bcjr_bit_errors += bcjr_errs

        fer = num_errors / num_iterations
        ber = num_bit_errors / (num_iterations * ldpc_code.n)
        bcjr_fer = num_bcjr_errors / num_iterations
        bcjr_ber = num_bcjr_bit_errors / (num_iterations * ldpc_code.n)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
        print(
            now_str
            + f" - Iteration {i + 1}: Errors: {errs}, FER: {fer}, BER: {ber} (BCJR-FER: {bcjr_fer}, BCJR-BER: {bcjr_ber})"
        )

    return (
        fer,
        ber,
        bcjr_fer,
        bcjr_ber,
        num_errors,
        num_bit_errors,
        num_bcjr_errors,
        num_bcjr_bit_errors,
    )


class SequenceGenerator:
    def __init__(
        self,
        iterations: int,
        A: int,
        ldpc_code: LinearCode,
        marker_code: MarkerCode,
        ids_channel: IDSChannel,
    ):
        self.iterations = iterations
        self.A = A
        self.ldpc_code = ldpc_code
        self.marker_code = marker_code
        self.ids_channel = ids_channel

        self._transmit = self.ids_channel.numba_compile_transmit()

    def __len__(self):
        return self.iterations

    def __iter__(self):
        for _ in range(self.iterations):
            ldpc_offset = np.random.randint(0, self.ldpc_code.q, self.ldpc_code.n)

            enc = self.marker_code.encode(ldpc_offset)

            r, N = self._transmit(enc, self.A)

            yield r, N, ldpc_offset


def single_transmission_multi(input, single_transmission_clb: Callable):
    r, N, ldpc_offset = input
    _, errs, _, bcjr_errs = single_transmission_clb(r, N, ldpc_offset)
    return errs, bcjr_errs, mp.current_process().pid


def run_simulation_multi_process(
    chunksize: int,
    workers: int,
    max_tasks_per_child: int | None,
    A: int,
    iterations: int,
    joint_decoding: bool,
    ldpc_bp_iterations: int,
    max_insertions: int,
    ids_channel: IDSChannel,
    ldpc_code: LinearCode,
    ldpc_gf: GaloisField2m,
    marker_code: MarkerCode,
    conv_code: MarkerConvolutionalCode,
):

    single_transmission_clb = partial(
        single_transmission_marker,
        joint_decoding=joint_decoding,
        ldpc_code=ldpc_code,
        marker_code=marker_code,
        conv_code=conv_code,
        ids_channel=ids_channel,
        ldpc_max_iterations=ldpc_bp_iterations,
        max_insertions=max_insertions,
        ldpc_gf=ldpc_gf,
    )

    single_transmission_multi_partial = partial(
        single_transmission_multi,
        single_transmission_clb=single_transmission_clb,
    )

    sequence_generator = SequenceGenerator(
        iterations, A, ldpc_code, marker_code, ids_channel
    )
    num_iterations = num_errors = num_bit_errors = num_bcjr_errors = (
        num_bcjr_bit_errors
    ) = 0

    chunksize = max(iterations // workers // 5, 1) if chunksize == -1 else chunksize

    print("Chunksize:", chunksize)

    previous_worker_id = None

    # initialize to avoid unbounded variable typing
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S:%f")
    prev_errs = fer = ber = bcjr_fer = bcjr_ber = -1

    with mp.Pool(workers, maxtasksperchild=max_tasks_per_child) as pool:
        for result in pool.imap_unordered(
            single_transmission_multi_partial,
            sequence_generator,
            chunksize=chunksize,
        ):
            errs, bcjr_errs, child_worker_pid = result
            if (
                previous_worker_id is not None
                and previous_worker_id != child_worker_pid
            ):
                log_msg = (
                    now_str
                    + f" - Iteration {num_iterations}/{iterations}: Errors: {prev_errs}, FER: {fer}, BER: {ber} (BCJR-FER: {bcjr_fer}, BCJR-BER: {bcjr_ber}) - m-pid: {mp.current_process().pid}, c-pid: {previous_worker_id}"
                )
                print(log_msg)

            now = datetime.now()

            num_iterations += 1
            num_errors += errs > 0
            num_bit_errors += errs
            num_bcjr_errors += bcjr_errs > 0
            num_bcjr_bit_errors += bcjr_errs

            fer = num_errors / num_iterations
            ber = num_bit_errors / (num_iterations * ldpc_code.n)
            bcjr_fer = num_bcjr_errors / num_iterations
            bcjr_ber = num_bcjr_bit_errors / (num_iterations * ldpc_code.n)

            now_str = now.strftime("%Y-%m-%d %H:%M:%S:%f")

            previous_worker_id = child_worker_pid
            prev_errs = errs

    log_msg = (
        now_str
        + f" - Iteration {num_iterations}/{iterations}: Errors: {prev_errs}, FER: {fer}, BER: {ber} (BCJR-FER: {bcjr_fer}, BCJR-BER: {bcjr_ber}) - m-pid: {mp.current_process().pid}, c-pid: {previous_worker_id}"
    )
    print(log_msg)

    return (
        fer,
        ber,
        bcjr_fer,
        bcjr_ber,
        num_errors,
        num_bit_errors,
        num_bcjr_errors,
        num_bcjr_bit_errors,
    )


@app.command(name="bcjr_bp")
def main(
    A: int = typer.Option(2, help="Number of transmissions"),
    q: int = typer.Option(2, help="Alphabet size"),
    iterations: int = typer.Option(50000, help="Number of transmissions"),
    joint_decoding: bool = typer.Option(
        False, help="Whether to use jointly decode all sequences or separately"
    ),
    ldpc_n: int = typer.Option(273, help="Number of bits in the input message"),
    ldpc_k: int = typer.Option(191, help="Number of bits in the output message"),
    ldpc_bp_iterations: int = typer.Option(
        30, help="Number of BP iterations for the outer LDPC"
    ),
    ldpc_q: int = typer.Option(2, help="Alphabet size for LDPC"),
    ldpc_file_extension: CodeFileExtension = typer.Option(
        CodeFileExtension.ALIST, help="File extension for LDPC code"
    ),
    ldpc_custom_file_name: str | None = typer.Option(
        None, help="Custom file name for LDPC code"
    ),
    ldpc_random_weights: bool = typer.Option(
        False,
        help="Whether to use random weights for the LDPC code (Only relevant for ldpc_q > 2)",
    ),
    marker: list[int] = typer.Option([0, 0, 1], help="Marker for LDPC code"),
    N_c: int = typer.Option(9, help="Number of bits between insertions of the marker"),
    p_i: float = typer.Option(0.012, help="Insertion probability"),
    p_d: float = typer.Option(0.012, help="Deletion probability"),
    p_s: float = typer.Option(0.004, help="Substitution probability"),
    max_insertions: int = typer.Option(
        2, help="Maximum number of insertions per symbol"
    ),
    custom_suffix: str | None = typer.Option(
        None, help="Custom suffix for output file"
    ),
    custom_output_path: Path | None = typer.Option(
        None, help="Output path for the results file"
    ),
    workers: int = typer.Option(
        0,
        help="Number of workers for parallel processing. If 0, no multiprocessing is started",
    ),
    chunksize: int = typer.Option(
        -1,
        help="Chunksize for multiprocessing. If -1, chunksize is chosen heuristically",
    ),
    max_tasks_per_child: int | None = typer.Option(
        None, help="Maximum number of tasks per child process"
    ),
):
    """Entrypoint for the BCJR simulation with LDPC and Marker codes"""

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")

    print("Start time:", start_time)

    cpus_avail = mp.cpu_count()

    print("Current Process ID:", mp.current_process().pid)
    print("Number of CPUs available:", cpus_avail)

    if workers > cpus_avail:
        logging.warning(
            f"Number of workers ({workers}) is greater than number of CPUs available ({cpus_avail}). Setting number of workers to {cpus_avail}"
        )
        workers = cpus_avail

    if workers > iterations:
        logging.warning(
            f"Number of workers ({workers}) is greater than number of iterations ({iterations}). Setting number of workers to {iterations}"
        )
        workers = iterations

    ids_channel = IDSChannel(
        q=q,
        p_i=p_i,
        p_d=p_d,
        p_s=p_s,
    )

    assert (
        np.log2(ldpc_q) % 1 == 0
    ), f"Only field 2^n are supported for ldpc q (input q: {ldpc_q})"

    ldpc_code = LinearCode.from_code_args(
        LinearCodeType.LDPC,
        ldpc_n,
        ldpc_k,
        ldpc_q,
        custom_file_name=ldpc_custom_file_name,
        code_file_extension=ldpc_file_extension,
        random_weights=ldpc_random_weights,
    )

    ldpc_gf = GaloisField2m(ldpc_q)

    assert np.log2(q) % 1 == 0, f"Only field 2^n are supported for q (input q: {q})"

    p = q.bit_length() - 1

    marker_code = MarkerCode(
        marker=marker,
        N_c=N_c,
        p=p,
        T=ldpc_n,
    )

    conv_code = MarkerConvolutionalCode(
        p, marker_code.encoded_length, random_offset=False
    )

    if workers > 0:
        (
            fer,
            ber,
            bcjr_fer,
            bcjr_ber,
            num_errors,
            num_bit_errors,
            num_bcjr_errors,
            num_bcjr_bit_errors,
        ) = run_simulation_multi_process(
            chunksize,
            workers,
            max_tasks_per_child,
            A,
            iterations,
            joint_decoding,
            ldpc_bp_iterations,
            max_insertions,
            ids_channel,
            ldpc_code,
            ldpc_gf,
            marker_code,
            conv_code,
        )
    else:
        (
            fer,
            ber,
            bcjr_fer,
            bcjr_ber,
            num_errors,
            num_bit_errors,
            num_bcjr_errors,
            num_bcjr_bit_errors,
        ) = run_simulation_single_process(
            A,
            iterations,
            joint_decoding,
            ldpc_bp_iterations,
            max_insertions,
            ids_channel,
            ldpc_code,
            ldpc_gf,
            marker_code,
            conv_code,
        )

    out_dict = {
        "A": A,
        "joint": joint_decoding,
        "iterations": iterations,
        "fer": fer,
        "ber": ber,
        "bcjr_fer": bcjr_fer,
        "bcjr_ber": bcjr_ber,
        "iterations": iterations,
        "ldpc_n": ldpc_code.n,
        "ldpc_k": ldpc_code.k,
        "marker": marker,
        "N_c": N_c,
        "q": q,
        "p_i": p_i,
        "p_d": p_d,
        "p_s": p_s,
        "max_insertions": max_insertions,
        "ldpc_max_iterations": ldpc_bp_iterations,
        "num_errors": int(num_errors),
        "num_bit_errors": int(num_bit_errors),
        "num_bcjr_errors": int(num_bcjr_errors),
        "num_bcjr_bit_errors": int(num_bcjr_bit_errors),
        "start_time": start_time,
        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f"),
    }

    out_file_base_name = "output_ldpc_{ldpc_n}_{ldpc_k}_A_{A}_joint_{joint}_marker_{marker}_N_c_{N_c}_q_{q}_iter_{iterations}_p_i_{p_i}_p_d_{p_d}_p_s_{p_s}_max_insertions_{max_insertions}_ldpc_max_iterations_{ldpc_max_iterations}{custom_suffix}".format(
        **out_dict, custom_suffix=("_" + custom_suffix if custom_suffix else "")
    )

    # We don't want to overwrite duplicate files => We might want to aggregate them afterwards if they have the same parameters
    i = 1
    no_out_file_base_name = out_file_base_name
    while os.path.exists(no_out_file_base_name + ".json"):
        no_out_file_base_name = out_file_base_name + f"_no_{i}"
        i += 1

    out_file = no_out_file_base_name + ".json"

    output_path = (
        out_file if custom_output_path is None else str(custom_output_path / out_file)
    )

    with open(output_path, "w") as f:
        json.dump(out_dict, f, indent=4)

    print("Wrote output to: ", out_file)


if __name__ == "__main__":

    app()
