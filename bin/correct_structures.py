import json
import os
import argparse
import logging
import shutil
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from biotite.structure import filter_backbone

import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from tqdm import tqdm

from bin.correct_structure import prepare_output_dir, download_model, get_real_len_of_structure, compute_pad_len, \
    read_to_correct_structure, overwrite_the_angles, load_missing_info_mask, fine_tune_predictions
from bin.sample import build_datasets, plot_ramachandran, SEED, \
    write_corrected_structures, generate_raports
from bin.structure_utils import mock_missing_info, determine_quality_of_structure, read_pdb_file, \
    gradient_descent_on_physical_constraints, write_structure_to_pdb

from foldingdiff import modelling
from foldingdiff import sampling
from foldingdiff.datasets import NoisedAnglesDataset, CathCanonicalAnglesOnlyDataset
from foldingdiff.angles_and_coords import canonical_distances_and_dihedrals, EXHAUSTIVE_ANGLES, \
    combine_original_with_predicted_structure
from foldingdiff import utils
from foldingdiff.sampling import sample_missing_structures


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
        "--input_dir",
        type=str,
        default="",
        help="Path to PDB file to correct",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=os.getcwd() + "/pdb_corrected",
        help="Path to output directory"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=512,
        help="Batch size to use when sampling. 256 consumes ~2GB of GPU memory, 512 ~3.5GB",
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
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force overwriting of output directory"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Make sure the input is valid
    assert args.input_dir != "", "Please provide path to PDB dir to correct"
    assert os.path.exists(args.input_dir), "Provided path to PDB dir to correct does not exist"
    assert os.path.isdir(args.input_dir), "Provided path to PDB dir to correct is not a directory"

    # Prepare output dir
    output_dir = prepare_output_dir(args)

    # Prepare device
    device = torch.device(args.device)

    # Download and load the model
    download_model(args)
    model = modelling.BertForDiffusionBase.from_dir(args.model).to(device)

    # Load the dataset
    train_dset, _, test_dset = build_datasets(
        Path(args.model), load_actual=False
    )

    # Get a list of all the PDB files in the input directory
    pdb_file_names = [
        f for f in os.listdir(args.input_dir) if f.endswith(".pdb")
    ]

    pdb_files_path = [
        os.path.join(args.input_dir, f)
        for f in pdb_file_names
    ]

    # Mock the pdb to correct files
    mocked_pdb_dir = str(output_dir / "mocked_pdb")
    os.makedirs(mocked_pdb_dir, exist_ok=True)
    mocked_pdb_files_path = [
        os.path.join(mocked_pdb_dir, f)
        for f in pdb_file_names
    ]

    progress_bar = tqdm(
        zip(pdb_files_path, mocked_pdb_files_path),
        total=len(pdb_files_path),
        desc="Mocking missing info"
    )

    for pdb_file_path, mocked_pdb_file_path in progress_bar:
        mock_missing_info(pdb_file_path, mocked_pdb_file_path)

    missing_residues_files = [
        mocked_pdb_file_path + ".missing"
        for mocked_pdb_file_path in mocked_pdb_files_path
    ]

    # Load the structures to correct
    real_lens = [
        get_real_len_of_structure(mocked_pdb_file_path)
        for mocked_pdb_file_path in mocked_pdb_files_path
    ]

    pad_len = compute_pad_len(
        max(real_lens),
        args.window_size,
        args.window_step
    )

    to_correct_features = [
        read_to_correct_structure(mocked_pdb_file_path, pad_len)
        for mocked_pdb_file_path in mocked_pdb_files_path
    ]

    to_correct_features = [
        overwrite_the_angles(features, mocked_pdb_file_path, train_dset, pad_len)
        for features, mocked_pdb_file_path in zip(to_correct_features, mocked_pdb_files_path)
    ]

    infill_masks = torch.vstack([
        load_missing_info_mask(missing_residues_file, features["attn_mask"])
        for missing_residues_file, features in zip(missing_residues_files, to_correct_features)
    ])

    angles = torch.vstack([
        features["angles"]
        for features in to_correct_features
    ])

    pad_masks = torch.vstack([
        features["attn_mask"]
        for features in to_correct_features
    ])

    # Perform correction
    torch.manual_seed(args.seed)
    corrected_angles = sample_missing_structures(
        model,
        train_dset,
        real_lens,
        infill_masks,
        angles,
        pad_masks,
        args.batch_size,
        window_step=args.window_step,
        window_size=args.window_size,
        whole_pad_len=pad_len,
    )

    sampled_dfs = [
        pd.DataFrame(s, columns=train_dset.feature_names["angles"])
        for s in corrected_angles
    ]

    # Write the raw sampled items to csv files
    sampled_angles_folder = output_dir / "sampled_corrected"
    os.makedirs(sampled_angles_folder, exist_ok=True)
    logging.info(f"Writing sampled angles to {sampled_angles_folder}")
    for i, s in enumerate(sampled_dfs):
        s.to_csv(sampled_angles_folder / f"{pdb_file_names[i]}.csv")

    # Read the original atom arrays
    original_atom_arrays = [
        read_pdb_file(mocked_pdb_file_path)
        for mocked_pdb_file_path in mocked_pdb_files_path
    ]

    # Write the corrected structures to PDB files
    corrected_structures_folder = output_dir / "corrected_structures"
    corrected_pdb_files = write_corrected_structures(
        final_sampled=sampled_dfs,
        output_dir=corrected_structures_folder,
        original_atom_array=original_atom_arrays,
        to_correct_mask=infill_masks,
        output_names=pdb_file_names,
    )

    # fine tune the structures with physical constraints
    fine_tuned_pdb_files = fine_tune_predictions(device, output_dir, corrected_pdb_files, infill_masks)



if __name__ == "__main__":
    main()