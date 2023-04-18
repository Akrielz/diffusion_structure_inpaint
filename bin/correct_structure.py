import gzip
import json
import os
import argparse
import logging
import shutil
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader

from bin.sample import build_datasets, plot_ramachandran, SEED, \
    write_corrected_structures, generate_raports
from bin.structure_utils import mock_missing_info, get_lowest_residue, determine_quality_of_structure, read_pdb_file

from foldingdiff import modelling
from foldingdiff import sampling
from foldingdiff.datasets import NoisedAnglesDataset, CathCanonicalAnglesOnlyDataset
from foldingdiff.angles_and_coords import canonical_distances_and_dihedrals, EXHAUSTIVE_ANGLES
from foldingdiff import utils


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser
    """
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="wukevin/foldingdiff_cath",
        help="Path to model directory, or a repo identifier on huggingface hub. Should contain training_args.json, config.json, and models folder at a minimum.",
    )
    parser.add_argument(
        "--outdir", "-o", type=str, default=os.getcwd(), help="Path to output directory"
    )
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=10,
        help="Number of examples to generate *per length*",
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=512,
        help="Batch size to use when sampling. 256 consumes ~2GB of GPU memory, 512 ~3.5GB",
    )
    parser.add_argument(
        "--fullhistory",
        action="store_true",
        help="Store full history, not just final structure",
    )
    parser.add_argument(
        "--testcomparison",
        action="store_true",
        help="Run comparison against test set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
    )
    parser.add_argument(
        "--pdb_to_correct",
        type=str,
        default="",
        help="Path to PDB file to correct",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=128,
        help="Minimum padding length that is needed"
    )
    parser.add_argument(
        "--window_step",
        type=int,
        default=32,
        help="Step size for the sliding window when correcting the structure",
    )
    return parser


def read_to_correct_structure(pdb_file: str, pad_len=128) -> Dict[str, torch.Tensor]:
    clean_dset = CathCanonicalAnglesOnlyDataset(
        pad=pad_len,
        trim_strategy='',
        fnames=[pdb_file],
        use_cache=True,
    )
    noised_dset = NoisedAnglesDataset(
        clean_dset,
        timesteps=1000,
        beta_schedule='cosine'
    )
    dl = DataLoader(noised_dset, batch_size=32, shuffle=False)
    features = iter(dl).next()

    return features


def get_real_len_of_structure(
        features: Union[Dict[str, torch.Tensor], str]
) -> int:
    # check if features is a Dict
    if isinstance(features, dict):
        attn_mask = features['attn_mask']
        real_len = torch.where(attn_mask == 1)[1].max().item() + 1
        return real_len

    # check if features is a str
    if isinstance(features, str):
        structure = read_pdb_file(features)
        real_len = len(structure[structure.atom_name == "CA"])
        return real_len

    raise ValueError("features should be either a Dict or a str")


def compute_angles_from_pdb(pdb_file: str):
    df = canonical_distances_and_dihedrals(pdb_file, angles=EXHAUSTIVE_ANGLES)
    return df


def overwrite_the_angles(
        to_correct_features: Dict[str, torch.Tensor],
        pdb_file: str,
        train_dset,
        pad_len: int = 128
):
    # Read the angles
    angles = compute_angles_from_pdb(pdb_file)
    angles = angles.to_numpy()
    angles = torch.from_numpy(angles)
    angles = angles.unsqueeze(0)

    # Shift things towards min:
    angles = angles - train_dset.dset.get_masked_means()
    angles = utils.modulo_with_wrapped_range(
        angles, range_min=-np.pi, range_max=np.pi
    )

    # Pad the angles
    len_to_pad = pad_len - angles.shape[1]
    angles = torch.cat(
        [angles, torch.zeros([1, len_to_pad, 6])], dim=1
    )

    # cast to float
    angles = angles.float()

    # Replaces nans with 0
    angles = torch.where(torch.isnan(angles), torch.zeros_like(angles), angles)

    to_correct_features["angles"] = angles
    return to_correct_features


def mock_missing_info_mask(features: Dict[str, torch.Tensor], num_missing=2) -> torch.Tensor:
    attn_mask = features['attn_mask']
    # select num_missing random positions that are masked
    masked_positions = torch.where(attn_mask == 1)[1]
    num_masked = len(masked_positions)

    # random_pos = torch.randperm(num_masked)[:num_missing]
    random_pos = torch.tensor([i for i in range(6, 8)])

    # create mask
    mask = torch.zeros_like(attn_mask)
    mask[:, masked_positions[random_pos]] = 1

    # mask = torch.zeros((1, 128))
    return mask


def load_missing_info_mask(
        missing_info_mask_file: str,
        attention_mask: torch.Tensor
) -> torch.Tensor:

    # Load the json file
    with open(missing_info_mask_file, "r") as f:
        missing_info_mask: Dict = json.load(f)

    # Extract the missing_residues_id
    missing_residues_id = missing_info_mask["missing_residues_id"]
    start_indexes = missing_info_mask["start_indexes"]

    chains = list(missing_residues_id.keys())
    for chain in chains:
        start_index = start_indexes[chain]
        missing_residues_id[chain] = [i - start_index for i in missing_residues_id[chain]]

    # Create the mask
    mask = torch.zeros_like(attention_mask)
    for chain in chains:
        mask[:, missing_residues_id[chain]] = 1

    return mask


def compute_pad_len(real_len, window_size, window_step):
    # We must have the pad len at least the window_size due to the model
    pad_len = max(window_size, real_len)

    # We must have the pad len divisible by the window_step
    pad_len = pad_len + (window_step - pad_len % window_step)

    return pad_len


def main():
    parser = build_parser()
    args = parser.parse_args()

    assert args.pdb_to_correct != "", "Please specify a PDB file to correct"

    logging.info(f"Creating {args.outdir}")
    os.makedirs(args.outdir, exist_ok=True)
    outdir = Path(args.outdir)

    if os.listdir(outdir):
        shutil.rmtree(outdir)
        os.makedirs(outdir, exist_ok=True)

    # Download the model if it was given on modelhub
    download_model(args)

    plotdir = outdir / "plots"
    os.makedirs(plotdir, exist_ok=True)

    # Load the dataset based on training args
    train_dset, _, test_dset = build_datasets(
        Path(args.model), load_actual=args.testcomparison
    )
    phi_idx = test_dset.feature_names["angles"].index("phi")
    psi_idx = test_dset.feature_names["angles"].index("psi")
    # Fetch values for training distribution
    select_by_attn = lambda x: x["angles"][x["attn_mask"] != 0]

    test_values_stacked = compute_ramachandran_plot(
        args, phi_idx, plotdir, psi_idx, select_by_attn, test_dset
    )

    # Mock the pdb to correct file
    mocked_pdb_file_path = str(outdir / "mocked_pdb/mocked.pdb")
    os.makedirs(outdir / "mocked_pdb", exist_ok=True)
    mock_missing_info(args.pdb_to_correct, mocked_pdb_file_path)
    missing_residues_file = mocked_pdb_file_path + ".missing"

    # Load the structure to correct
    to_correct_real_len = get_real_len_of_structure(mocked_pdb_file_path)
    pad_len = compute_pad_len(to_correct_real_len, args.window_size, args.window_step)
    to_correct_features = read_to_correct_structure(mocked_pdb_file_path, pad_len)
    to_correct_features = overwrite_the_angles(to_correct_features, mocked_pdb_file_path, train_dset, pad_len)
    to_correct_mask = load_missing_info_mask(missing_residues_file, to_correct_features["attn_mask"])

    # Load the model
    model_snapshot_dir = outdir / "model_snapshot"
    model = modelling.BertForDiffusionBase.from_dir(
        args.model, copy_to=model_snapshot_dir
    ).to(torch.device(args.device))

    # Perform sampling
    torch.manual_seed(args.seed)
    sampled = sampling.sample_missing_structure(
        model,
        train_dset,
        to_correct_real_len,
        to_correct_mask,
        to_correct_features,
        n=args.num,
        batch_size=args.batchsize,
        window_step=args.window_step,
        window_size=args.window_size,
        whole_pad_len=pad_len,
    )

    final_sampled = [s for s in sampled]
    sampled_dfs = [
        pd.DataFrame(s, columns=train_dset.feature_names["angles"])
        for s in final_sampled
    ]

    # Write the raw sampled items to csv files
    sampled_angles_folder = outdir / "sampled_angles"
    os.makedirs(sampled_angles_folder, exist_ok=True)
    logging.info(f"Writing sampled angles to {sampled_angles_folder}")
    for i, s in enumerate(sampled_dfs):
        s.to_csv(sampled_angles_folder / f"generated_{i}.csv.gz")

    # read the atom_array of the structure to correct
    to_correct_atom_array = read_pdb_file(mocked_pdb_file_path)

    # Write the sampled angles as pdb files
    pdb_files = write_corrected_structures(sampled_dfs, outdir / "sampled_pdb", to_correct_atom_array, to_correct_mask)

    generate_raports(args, final_sampled, outdir, pdb_files, phi_idx, plotdir, psi_idx, sampled, sampled_angles_folder,
                     test_dset, test_values_stacked, train_dset)

    # Iterate through the generated structures and compute their quality
    scores = [determine_quality_of_structure(read_pdb_file(pdb_file)) for pdb_file in pdb_files]

    # Get the lowest score / best structure
    best_structure = pdb_files[np.argmin(scores)]

    # Write the best structure to a file
    # Make the directory for the best
    os.makedirs(outdir / "best_pdb", exist_ok=True)
    shutil.copy(best_structure, outdir / "best_pdb/best.pdb")


def download_model(args):
    if utils.is_huggingface_hub_id(args.model):
        logging.info(f"Detected huggingface repo ID {args.model}")
        dl_path = snapshot_download(args.model)  # Caching is automatic
        assert os.path.isdir(dl_path)
        logging.info(f"Using downloaded model at {dl_path}")
        args.model = dl_path


def compute_ramachandran_plot(args, phi_idx, plotdir, psi_idx, select_by_attn, test_dset):
    if args.testcomparison:
        test_values = [
            select_by_attn(test_dset.dset.__getitem__(i, ignore_zero_center=True))
            for i in range(len(test_dset))
        ]
        test_values_stacked = torch.cat(test_values, dim=0).cpu().numpy()

        # Plot ramachandran plot for the training distribution
        # Default figure size is 6.4x4.8 inches
        plot_ramachandran(
            test_values_stacked[:, phi_idx],
            test_values_stacked[:, psi_idx],
            annot_ss=True,
            fname=plotdir / "ramachandran_test_annot.pdf",
        )
    else:
        test_values_stacked = None

    return test_values_stacked


if __name__ == '__main__':
    main()