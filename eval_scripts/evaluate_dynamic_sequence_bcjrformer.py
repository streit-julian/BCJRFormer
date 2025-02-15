import argparse
import datetime
import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bcjrformer.codes.linear_code import CodeFileExtension, LinearCode, LinearCodeType
from bcjrformer.codes.marker_code import MarkerCode
from bcjrformer.configs.model_config import ModelConfig
from bcjrformer.datasets.bcjrformer_dataset import BCJRFormerMarkerDataset
from bcjrformer.models.bcjrformer import BCJRFormer
from bcjrformer.trainers.bcjrformer_marker_trainer import BCJRFormerMarkerTrainer
from bcjrformer.utils import repair_compiled_state_dict

logging.basicConfig(level=logging.INFO)


def evaluate_dynamic_model(
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

    model_config = model_dict["config"]

    model_state_dict = model_dict["model_state_dict"]
    if model_config.compile_model:
        model_state_dict = repair_compiled_state_dict(model_state_dict)

    n_sequence_min = model_config.n_sequence_min
    n_sequence_max = model_config.n_sequence_max
    std_mult = model_config.inner_model_std_multiplier
    channel_config = model_config.channel

    code = LinearCode.from_code_args(
        code_type=LinearCodeType.LDPC,
        n=n,
        k=k,
        q=2,
        code_file_extension=CodeFileExtension.TXT,
    )

    marker_code = MarkerCode(marker=marker, N_c=N_c, p=1, T=n)

    window_size = model_state_dict["to_patch_embedding.1.weight"].shape[1]

    bcjrformer = BCJRFormer(
        model_config,
        window_size,
        marker_code.encoded_length,
        device,
    )

    bcjrformer.load_state_dict(model_state_dict)

    bcjrformer.to(device)

    base_config_dict = model_config.__dict__

    if "separate_checkpoints" in base_config_dict:
        base_config_dict.pop("separate_checkpoints")

    base_config_no_log = ModelConfig(**{**base_config_dict, "log_wandb": False})

    bcjrformer_marker_trainer = BCJRFormerMarkerTrainer(
        model_config=base_config_no_log,
        window_block_dimension=window_size,
        marker_code=marker_code,
        device=device,
    )

    bcjrformer_marker_trainer.model = bcjrformer
    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a descriptive file name
    file_name = f"evaluate_p_d_{channel_config.p_d}_p_i_{channel_config.p_i}_p_s_{channel_config.p_s}_n_sequence_{n_sequence_min}_{n_sequence_max}_std_mult_{std_mult}_timestamp_{timestamp}.json"

    evaluation_data = []

    for seq_length in range(n_sequence_min, n_sequence_max + 1):
        logging.info(
            f"----------------- Evaluating Sequence length: {seq_length} -----------------"
        )
        dataset = BCJRFormerMarkerDataset(
            code,
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
            fixed_n_sequence=seq_length,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=workers,
        )

        test_loss, log_dict = bcjrformer_marker_trainer.evaluate(dataloader)

        eval_data = {
            **log_dict,
            "sequence_length": seq_length,
            "test_loss": test_loss,
        }

        evaluation_data.append(eval_data)
        logging.info(eval_data)

        logging.info(
            f"----------------- Finished Evaluating Sequence length: {seq_length} -----------------"
        )

    evaluation_dict = {
        "model_path": str(model_path),
        "start_timestamp": timestamp,
        "end_timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "evaluation_data": evaluation_data,
    }

    with open(Path("./results") / file_name, "w") as f:
        json.dump(evaluation_dict, f, indent=4)
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
        evaluate_dynamic_model(
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
