import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bcjrformer.codes.galois_field import GaloisField2m
from bcjrformer.codes.linear_code import CodeFileExtension, LinearCode, LinearCodeType
from bcjrformer.codes.marker_code import MarkerCode
from bcjrformer.datasets.bcjrformer_dataset import BCJRFormerMarkerDataset
from bcjrformer.models.bcjrformer import BCJRFormer
from bcjrformer.utils import repair_compiled_state_dict

logging.basicConfig(level=logging.INFO)


def evaluate_inner_bcjrformer_outer_bp(
    model_directory: str,
    n: int,
    k: int,
    marker: list[int],
    N_c: int,
    test_batch_size: int = 256,
    test_batches_per_epoch: int = 1600,
    workers: int = 48,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_name = "model_IDS_ECCT_CONCATENATED_MARKER_INNER_IDSChannelConfig(p_i=0.005, p_d=0.005, p_s=0.0, fixed_del_count=None)_LDPC_n_96_k_48_MarkerCodeConfig(marker=[0, 0, 1], N_c=6, p=1)_05_01_2025_14_01_42__low_complex_a1_v2"
    # base_path = "./Results_IDS_ECCT_CONCATENATED_MARKER_INNER"
    model = "best_model.pth"
    # model_path = Path(base_path) / model_name / model

    model_path = Path(model_directory) / model

    assert model_path.exists(), f"Model path {model_path} does not exist"

    model_dict = torch.load(model_path)

    base_config = model_dict["config"]

    model_state_dict = model_dict["model_state_dict"]
    if base_config.compile_model:
        model_state_dict = repair_compiled_state_dict(model_state_dict)

    n_sequence_min = base_config.n_sequence_min
    n_sequence_max = base_config.n_sequence_max
    std_mult = base_config.inner_model_std_multiplier
    channel_config = base_config.channel

    linear_code = LinearCode.from_code_args(
        code_type=LinearCodeType.LDPC,
        n=n,
        k=k,
        q=2,
        code_file_extension=CodeFileExtension.ALIST,
    )

    marker_code = MarkerCode(marker=marker, N_c=N_c, p=1, T=n)

    window_size = model_state_dict["to_patch_embedding.1.weight"].shape[1]

    inner_model = BCJRFormer(
        base_config,
        window_size,
        marker_code.encoded_length,
        device=device,
    )

    inner_model.load_state_dict(model_state_dict)

    inner_model.to(device)

    # base_config_no_log = BaseModelConfig(**{**base_config.__dict__, "log_wandb": False})

    logging.info("----------------- Evaluating BP on model -----------------")
    dataset = BCJRFormerMarkerDataset(
        linear_code,
        marker_code,
        p_i=channel_config.p_i,
        p_d=channel_config.p_d,
        p_s=channel_config.p_s,
        batch_size=test_batch_size,
        batches_per_epoch=test_batches_per_epoch,
        compare_bcjr=False,
        std_mult=std_mult,
        n_sequence_max=n_sequence_max,
        n_sequence_min=n_sequence_min,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=workers,
    )

    inner_model.eval()

    bp_max_iterations = 100
    gf = GaloisField2m(linear_code.q)
    cum_samples = 0
    cum_ber_inner = cum_fer_inner = cum_ber_bp = cum_fer_bp = 0.0
    with torch.no_grad():
        for n_batch, (
            m,
            x,
            x_i,
            _,
            y,
            # padding_mask,
            _,
        ) in enumerate(iter(dataloader)):
            x_inner = inner_model(y.to(device))

            x_inner_logits = torch.sigmoid(x_inner)

            # remove markers
            x_logits, _ = marker_code.decode_model(x_inner_logits)

            x_pred = torch.round(x_logits)

            # provide full output for BP
            x_full_logits = torch.concat(
                (1 - x_logits.unsqueeze(-1), x_logits.unsqueeze(-1)), dim=-1
            )

            # move to exponential domain (L_ei)
            x_log = torch.log(x_full_logits).cpu().numpy().astype(np.float64)

            x_dev = x.to(device)
            x_np = x.numpy()

            ber_inner = fer_inner = ber_bp = fer_bp = 0.0
            for i, (x_log_i, x_np_i) in enumerate(zip(x_log, x_np)):
                x_bp = linear_code.belief_propagation(
                    x_log_i, max_iter=bp_max_iterations, gf=gf
                )

                x_bp_pred = x_bp.argmax(axis=-1)

                ber_inner += (x_dev[i] != x_pred[i]).float().mean().item()
                fer_inner += (x_dev[i] != x_pred[i]).any().float().item()
                ber_bp += (x_np_i != x_bp_pred).mean()
                fer_bp += (x_np_i != x_bp_pred).any().astype(int)

            cum_samples += x.shape[0]
            cum_ber_inner += ber_inner
            cum_fer_inner += fer_inner
            cum_ber_bp += ber_bp
            cum_fer_bp += fer_bp

            logging.info(
                f"------------ Batch: {n_batch + 1}/{test_batches_per_epoch} -----------\n"
                + f"BER BP (BATCH): {ber_bp / test_batch_size} \n"
                + f"FER BP (BATCH): {fer_bp / test_batch_size} \n"
                + f"BER BP Total: {cum_ber_bp / cum_samples}   \n"
                + f"FER BP Total: {cum_fer_bp / cum_samples} \n"
                + f"BER Inner (BATCH):{ber_inner / test_batch_size} \n"
                + f"FER Inner (BATCH):{fer_inner / test_batch_size} \n"
                + f"BER Inner Total:{cum_ber_inner / cum_samples} \n"
                + f"FER Inner Total:{cum_fer_inner / cum_samples} \n"
                + "\n"
            )

    evaluation_data = {
        "N": n,
        "K": k,
        "Marker": marker_code.marker,
        "N_c": marker_code.N_c,
        "BER_BP": cum_ber_bp / cum_samples,
        "FER_BP": cum_fer_bp / cum_samples,
        "BER_Inner": cum_ber_inner / cum_samples,
        "FER_Inner": cum_fer_inner / cum_samples,
        "BP Iterations": bp_max_iterations,
        "CUM_SAMPLES": cum_samples,
        "MODEL_PATH": str(model_path),
    }

    file_name = f"evaluate_bp_N_{n}_k_{k}_pi_{channel_config.p_i}_pd_{channel_config.p_d}_ps_{channel_config.p_s}_marker_{marker_code.marker}_Nc_{marker_code.N_c}_std_mult_{std_mult}_bp_iter_{bp_max_iterations}.json"

    with open(Path("./results") / file_name, "w") as f:
        json.dump(evaluation_data, f, indent=4)

    logging.info(f"Saved evaluation data to {file_name}")


def main(
    model_directories: list[str],
    n: int,
    k: int,
    marker: list[int],
    N_c: int,
    workers: int,
    gpu: int,
):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for model_directory in model_directories:
        evaluate_inner_bcjrformer_outer_bp(
            model_directory=model_directory,
            n=n,
            k=k,
            marker=marker,
            N_c=N_c,
            workers=workers,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_directories",
        nargs="+",
        help="Directory where model is stored",
    )

    parser.add_argument("--n", type=int, help="Length of the codeword", default=96)
    parser.add_argument("--k", type=int, help="Length of the message", default=48)
    parser.add_argument(
        "--marker", type=int, nargs="+", help="Marker sequence", default=[0, 0, 1]
    )
    parser.add_argument("--N_c", type=int, help="Number of markers", default=6)

    parser.add_argument("--workers", type=int, help="Number of workers", default=48)

    parser.add_argument("--gpu", type=int, help="Which gpu to use", default=0)

    args = parser.parse_args()

    main(
        model_directories=args.model_directories,
        n=args.n,
        k=args.k,
        marker=args.marker,
        N_c=args.N_c,
        workers=args.workers,
        gpu=args.gpu,
    )
