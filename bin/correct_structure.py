import gc
import json
import os
import argparse
import logging
import shutil
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd

import torch
from huggingface_hub import snapshot_download
from torch.nn.functional import pad
from tqdm import tqdm

from bin.sample import build_datasets, plot_ramachandran, SEED, \
    write_corrected_structures
from bin.structure_utils import determine_quality_of_structure, read_pdb_file, \
    gradient_descent_on_physical_constraints, write_structure_to_pdb, mock_missing_info_by_alignment, filter_backbone, \
    reindex_pdb_file, read_pdb_header, add_header_info

from foldingdiff import modelling
from foldingdiff import sampling
from foldingdiff.datasets import NoisedAnglesDataset
from foldingdiff.angles_and_coords import canonical_distances_and_dihedrals, EXHAUSTIVE_ANGLES, \
    combine_original_with_predicted_structure
from foldingdiff import utils


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser

    Returns:
        argparse.ArgumentParser: CLI parser
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
        "--output_dir",
        "-o",
        type=str,
        default=os.getcwd() + "/pdb_corrected",
        help="Path to output directory"
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
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force overwriting of output directory"
    )
    return parser


def get_real_len_of_structure(
        features: Union[Dict[str, torch.Tensor], str]
) -> int:
    """
    Get the real length of the structure

    Args:
        features (Union[Dict[str, torch.Tensor], str]): Either a Dict with the
            keys "attn_mask" or a str with the path to the PDB file

    Returns:
        int: The real length of the structure

    Raises:
        ValueError: If the features is neither a Dict nor a str
    """

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


def compute_angles_from_pdb(pdb_file: str) -> pd.DataFrame:
    """
    Compute the angles from the PDB file

    Args:
        pdb_file (str): Path to the PDB file

    Returns:
        pd.DataFrame: DataFrame with the angles
    """
    df = canonical_distances_and_dihedrals(pdb_file, angles=EXHAUSTIVE_ANGLES)
    return df


def overwrite_the_angles(
        to_correct_features: Dict[str, torch.Tensor],
        pdb_file: str,
        train_dset,
        pad_len: int = 128
):
    """
    Overwrite the angles in the to_correct_features with the angles from the
    pdb_file. This is relevant as these angles are shifted in the proper way
    to be handled by the model, and are also padded to the correct length.

    Args:
        to_correct_features (Dict[str, torch.Tensor]):
            Dict with the features to correct
        pdb_file (str):
            Path to the PDB file
        train_dset (NoisedAnglesDataset):
            Dataset used for training
        pad_len (int, optional):
            Length to pad the angles to. Defaults to 128.

    Returns:
        Dict[str, torch.Tensor]:
            Dict with the corrected features
    """

    # Read the angles
    angles = compute_angles_from_pdb(pdb_file)

    if angles is None:
        print("Could not compute angles from the structure")
        raise ValueError

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


def load_missing_info_mask(
        missing_info_mask_file: str,
        pad_len: int
) -> torch.Tensor:
    """
    Load the missing_info_mask from the missing_info_mask_file

    Args:
        missing_info_mask_file (str):
            Path to the missing_info_mask_file

        pad_len (int):
            Length of the mask

    Returns:
        torch.Tensor:
    """

    # Load the json file
    with open(missing_info_mask_file, "r") as f:
        missing_info_mask: Dict = json.load(f)

    # Extract the missing_residues_id
    missing_residues_id = missing_info_mask["missing_residues_id"]
    index_mapping = missing_info_mask["index_mapping"]

    chains = list(missing_residues_id.keys())
    for chain in chains:
        missing_residues_id[chain] = [
            index_mapping[chain][f"{i}"] - 1  # The reindex is starting from 1
            for i in missing_residues_id[chain]
        ]

    # Create the mask
    mask = torch.zeros([1, pad_len])
    for chain in chains:
        mask[:, missing_residues_id[chain]] = 1

    return mask


def compute_pad_len(real_len: int, window_size: int, window_step: int) -> int:
    """
    Compute the pad_len based on the real_len, window_size and window_step

    Args:
        real_len (int):
            Real length of the structure
        window_size (int):
            Window size of the model
        window_step (int):
            Window step of the model

    Returns:
        int:
            The pad_len

    Example:
        compute_pad_len(10, 38, 16) == 48
    """
    # We must have the pad len at least the window_size due to the model
    pad_len = max(window_size, real_len)

    # We must have the pad len divisible by the window_step
    pad_len = pad_len + (window_step - pad_len % window_step)

    return pad_len


def main():
    parser = build_parser()
    args = parser.parse_args()

    assert args.pdb_to_correct != "", "Please specify a PDB file to correct"

    output_dir = prepare_output_dir(args)

    # Download the model if it was given on modelhub
    download_model(args)

    plotdir = output_dir / "plots"
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
    mocked_pdb_dir = output_dir / "mocked_pdb"
    mocked_pdb_file_path = str(mocked_pdb_dir / "mocked.pdb")
    reindex_pdb_file_path = str(mocked_pdb_dir / "mocked_reindex.pdb")
    os.makedirs(output_dir / "mocked_pdb", exist_ok=True)
    mock_missing_info_by_alignment(args.pdb_to_correct, mocked_pdb_file_path)
    with open(mocked_pdb_file_path + ".missing", "r") as f:
        index_mapping = json.load(f)['index_mapping']
    reindex_pdb_file(mocked_pdb_file_path, reindex_pdb_file_path, index_mapping)
    missing_residues_file = mocked_pdb_file_path + ".missing"

    # Load the structure to correct
    real_len = get_real_len_of_structure(reindex_pdb_file_path)
    attention_mask = torch.ones([1, real_len])
    attention_mask[0, :real_len] = 1

    pad_len = compute_pad_len(real_len, args.window_size, args.window_step)
    to_correct_features = {
        "attn_mask": attention_mask,
    }

    # Compute angles
    to_correct_features = overwrite_the_angles(
        to_correct_features, reindex_pdb_file_path, train_dset, pad_len
    )

    to_correct_mask = load_missing_info_mask(missing_residues_file, pad_len)

    # Load the model
    model = modelling.BertForDiffusionBase.from_dir(
        args.model
    ).to(torch.device(args.device))

    # Perform sampling
    torch.manual_seed(args.seed)
    sampled = sampling.sample_missing_structure(
        model,
        train_dset,
        real_len,
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
    sampled_angles_folder = output_dir / "sampled_angles"
    os.makedirs(sampled_angles_folder, exist_ok=True)
    logging.info(f"Writing sampled angles to {sampled_angles_folder}")
    for i, s in enumerate(sampled_dfs):
        s.to_csv(sampled_angles_folder / f"generated_{i}.csv.gz")

    # read the atom_array of the structure to correct
    to_correct_atom_array = read_pdb_file(reindex_pdb_file_path)

    # Write the sampled angles as pdb files
    pdb_files = write_corrected_structures(sampled_dfs, output_dir / "sampled_pdb", to_correct_atom_array, to_correct_mask)
    device = torch.device(args.device)

    # Fine tune the sampled structures
    fine_tuned_pdb_files = fine_tune_predictions(device, output_dir, pdb_files, to_correct_mask)

    # generate_raports(args, final_sampled, output_dir, pdb_files, phi_idx, plotdir, psi_idx, sampled, sampled_angles_folder,
    #                  test_dset, test_values_stacked, train_dset)

    # Iterate through the generated structures and compute their quality
    determine_best_pdb("original_best.pdb", output_dir, pdb_files)
    determine_best_pdb("fine_tuned_best.pdb", output_dir, fine_tuned_pdb_files)

    # add the old header to the fine-tuned and corrected pdb files
    header = read_pdb_header(reindex_pdb_file_path)
    for i, (fine_tuned, corrected) in enumerate(zip(fine_tuned_pdb_files, pdb_files)):
        add_header_info(header, fine_tuned)
        add_header_info(header, corrected)


def prepare_output_dir(args):
    output_dir = Path(args.output_dir)
    if os.path.exists(output_dir):
        if args.force:
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"Output directory {output_dir} already exists. Use --force to overwrite")
    logging.info(f"Creating {output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    return output_dir


def determine_best_pdb(best_name, output_dir, pdb_files):
    scores = [determine_quality_of_structure(read_pdb_file(pdb_file)) for pdb_file in pdb_files]
    best_structure = pdb_files[np.argmin(scores)]
    os.makedirs(output_dir / "best_pdb", exist_ok=True)
    shutil.copy(best_structure, output_dir / f"best_pdb/{best_name}")


def fine_tune_predictions(
        device,
        output_dir,
        pdb_files,
        to_correct_mask,
        batch_size: int = 16,
):
    fine_tuned = []
    os.makedirs(output_dir / "fine_tuned", exist_ok=True)

    if to_correct_mask.shape[0] == 1:
        to_correct_mask = to_correct_mask.repeat(len(pdb_files), 1)

    all_coords = []
    all_atom_masks = []
    for pdb_file, mask in zip(pdb_files, to_correct_mask):
        # Read the pdb file
        structure = read_pdb_file(pdb_file)

        # Get the backbone atoms
        backbone_atoms = structure[filter_backbone(structure)]
        coords = backbone_atoms.coord
        coords = torch.tensor(coords, dtype=torch.float32)

        # Create atom mask
        residue_indexes = torch.where(mask == 1)[0]
        atom_mask = torch.zeros(coords.shape[0], dtype=torch.bool)
        for i in residue_indexes:
            atom_mask[3 * i:3 * i + 3] = True

        all_coords.append(coords)
        all_atom_masks.append(atom_mask)

        # Free memory
        gc.collect()
        torch.cuda.empty_cache()

    original_lens = [len(c) for c in all_coords]
    max_len = max(original_lens)

    # Pad the coords and masks
    pad_mask = [torch.zeros(max_len).bool() for _ in range(len(all_coords))]
    for i in range(len(all_coords)):
        all_coords[i] = pad(all_coords[i], (0, 0, 0, max_len - len(all_coords[i])))
        all_atom_masks[i] = pad(all_atom_masks[i], (0, max_len - len(all_atom_masks[i])))
        pad_mask[i][:original_lens[i]] = 1

    all_coords = torch.stack(all_coords)
    all_atom_masks = torch.stack(all_atom_masks)
    pad_mask = torch.stack(pad_mask)

    all_coords_batched = [
        all_coords[i:i + batch_size] for i in range(0, len(all_coords), batch_size)
    ]
    all_atom_masks_batched = [
        all_atom_masks[i:i + batch_size] for i in range(0, len(all_atom_masks), batch_size)
    ]
    pad_mask_batched = [
        pad_mask[i:i + batch_size] for i in range(0, len(pad_mask), batch_size)
    ]
    original_lens_batched = [
        original_lens[i:i + batch_size] for i in range(0, len(original_lens), batch_size)
    ]

    all_coords_updated = []
    for coords, atom_masks, mask, orig_len in \
            zip(all_coords_batched, all_atom_masks_batched, pad_mask_batched, original_lens_batched):

        crop_len = max(orig_len)

        coords_cropped = coords[:, :crop_len]
        atom_masks_cropped = atom_masks[:, :crop_len]
        mask_cropped = mask[:, :crop_len]

        sampled_coords = coords_cropped
        for lr in [0.1, 0.01, 0.001]:
            sampled_coords = gradient_descent_on_physical_constraints(
                coords=sampled_coords,
                inpaint_mask=atom_masks_cropped,
                pad_mask=mask_cropped,
                num_epochs=10000,
                stop_patience=300,
                show_progress=True,
                device=device,
                lr=lr,
            )

        sampled_coords = sampled_coords.cpu().numpy()

        coords[:, :crop_len] = torch.tensor(sampled_coords)
        all_coords_updated.append(coords.numpy())

    all_coords_updated = np.concatenate(all_coords_updated)

    progress_bar = tqdm(
        enumerate(pdb_files),
        total=len(pdb_files),
        desc="Writing fine tuned structures"
    )
    for i, pdb_file in progress_bar:

        # Read the pdb file
        structure = read_pdb_file(pdb_file)

        original_len = original_lens[i]

        coords = all_coords_updated[i][:original_len]

        replaced_info_mask = to_correct_mask.cpu()[i].numpy().astype(bool)
        replaced_info_mask = replaced_info_mask[:original_len]
        new_atom_array = combine_original_with_predicted_structure(
            original_atom_array=structure,
            replaced_info_mask=replaced_info_mask,
            nerf_coords=coords,
        )

        current_file_name = Path(pdb_file).name
        out_file_name = str(output_dir / "fine_tuned" / current_file_name)

        write_structure_to_pdb(new_atom_array, out_file_name)
        fine_tuned.append(out_file_name)

    return fine_tuned


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