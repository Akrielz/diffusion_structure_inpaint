import gzip
import multiprocessing
import os, sys
import argparse
import logging
import json
import shutil
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import mpl_scatter_density
from biotite.structure.io.pdb import PDBFile
from matplotlib import pyplot as plt
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import torch
from huggingface_hub import snapshot_download
from pandas import DataFrame
from torch.utils.data import DataLoader

from bin.sample import build_datasets, plot_ramachandran, SEED, write_preds_pdb_folder, FT_NAME_MAP, \
    plot_distribution_overlap, write_corrected_structures
# Import data loading code from main training script
from train import get_train_valid_test_sets
from annot_secondary_structures import make_ss_cooccurrence_plot

from foldingdiff import modelling
from foldingdiff import sampling
from foldingdiff import plotting
from foldingdiff.datasets import AnglesEmptyDataset, NoisedAnglesDataset, CathCanonicalAnglesOnlyDataset
from foldingdiff.angles_and_coords import create_new_chain_nerf_to_file, \
    canonical_distances_and_dihedrals, EXHAUSTIVE_ANGLES
from foldingdiff import utils


def read_pdb_file(fname: str):
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        source = PDBFile.read(f)
    if source.get_model_count() > 1:
        return None
    # Pull out the atomarray from atomarraystack
    source_struct = source.get_structure()[0]

    return source_struct


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
        "--testcomparison", action="store_true", help="Run comparison against test set"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument(
        "--pdb_to_correct",
        type=str,
        default="",
        help="Path to PDB file to correct",
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


def get_real_len_of_structure(features: Dict[str, torch.Tensor]) -> int:
    attn_mask = features['attn_mask']
    real_len = torch.where(attn_mask == 1)[1].max().item() + 1
    return real_len


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
    random_pos = torch.tensor([i for i in range(4, 16)])

    # create mask
    mask = torch.zeros_like(attn_mask)
    mask[:, masked_positions[random_pos]] = 1

    # mask = torch.zeros((1, 128))
    return mask


def main():
    parser = build_parser()
    args = parser.parse_args()

    assert args.pdb_to_correct != "", "Please specify a PDB file to correct"

    logging.info(f"Creating {args.outdir}")
    os.makedirs(args.outdir, exist_ok=True)
    outdir = Path(args.outdir)
    # Be extra cautious so we don't overwrite any results

    if os.listdir(outdir):
        # remove the directory and all its contents
        shutil.rmtree(outdir)
        # recreate the directory
        os.makedirs(outdir, exist_ok=True)

    # Download the model if it was given on modelhub
    if utils.is_huggingface_hub_id(args.model):
        logging.info(f"Detected huggingface repo ID {args.model}")
        dl_path = snapshot_download(args.model)  # Caching is automatic
        assert os.path.isdir(dl_path)
        logging.info(f"Using downloaded model at {dl_path}")
        args.model = dl_path

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

    # Load the structure to correct
    to_correct_features = read_to_correct_structure(args.pdb_to_correct)
    to_correct_features = overwrite_the_angles(to_correct_features, args.pdb_to_correct, train_dset)
    to_correct_mask = mock_missing_info_mask(to_correct_features, num_missing=4)
    to_correct_real_len = get_real_len_of_structure(to_correct_features)

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
    )

    final_sampled = [s[-1] for s in sampled]
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
    to_correct_atom_array = read_pdb_file(args.pdb_to_correct)

    # Write the sampled angles as pdb files
    pdb_files = write_corrected_structures(sampled_dfs, outdir / "sampled_pdb", to_correct_atom_array, to_correct_mask)
    # pdb_files = write_preds_pdb_folder(sampled_dfs, outdir / "sampled_pdb")

    # If full history is specified, create a separate directory and write those files
    if args.fullhistory:
        # Write the angles
        full_history_angles_dir = sampled_angles_folder / "sample_history"
        os.makedirs(full_history_angles_dir)
        full_history_pdb_dir = outdir / "sampled_pdb/sample_history"
        os.makedirs(full_history_pdb_dir)
        # sampled is a list of np arrays
        for i, sampled_series in enumerate(sampled):
            snapshot_dfs = [
                pd.DataFrame(snapshot, columns=train_dset.feature_names["angles"])
                for snapshot in sampled_series
            ]
            # Write the angles
            ith_angle_dir = full_history_angles_dir / f"generated_{i}"
            os.makedirs(ith_angle_dir, exist_ok=True)
            for timestep, snapshot_df in enumerate(snapshot_dfs):
                snapshot_df.to_csv(
                    ith_angle_dir / f"generated_{i}_timestep_{timestep}.csv.gz"
                )
            # Write the pdb files
            ith_pdb_dir = full_history_pdb_dir / f"generated_{i}"
            write_preds_pdb_folder(
                snapshot_dfs, ith_pdb_dir, basename_prefix=f"generated_{i}_timestep_"
            )

    # Generate histograms of sampled angles -- separate plots, and a combined plot
    # For calculating angle distributions
    multi_fig, multi_axes = plt.subplots(
        dpi=300, nrows=2, ncols=3, figsize=(14, 6), sharex=True
    )
    step_multi_fig, step_multi_axes = plt.subplots(
        dpi=300, nrows=2, ncols=3, figsize=(14, 6), sharex=True
    )
    final_sampled_stacked = np.vstack(final_sampled)
    for i, ft_name in enumerate(test_dset.feature_names["angles"]):
        orig_values = (
            test_values_stacked[:, i] if test_values_stacked is not None else None
        )
        samp_values = final_sampled_stacked[:, i]

        ft_name_readable = FT_NAME_MAP[ft_name]

        # Plot single plots
        plot_distribution_overlap(
            {"Test": orig_values, "Sampled": samp_values},
            title=f"Sampled angle distribution - {ft_name_readable}",
            fname=plotdir / f"dist_{ft_name}.pdf",
        )
        plot_distribution_overlap(
            {"Test": orig_values, "Sampled": samp_values},
            title=f"Sampled angle CDF - {ft_name_readable}",
            histtype="step",
            cumulative=True,
            fname=plotdir / f"cdf_{ft_name}.pdf",
        )

        # Plot combo plots
        plot_distribution_overlap(
            {"Test": orig_values, "Sampled": samp_values},
            title=f"Sampled angle distribution - {ft_name_readable}",
            ax=multi_axes.flatten()[i],
            show_legend=i == 0,
        )
        plot_distribution_overlap(
            {"Test": orig_values, "Sampled": samp_values},
            title=f"Sampled angle CDF - {ft_name_readable}",
            cumulative=True,
            histtype="step",
            ax=step_multi_axes.flatten()[i],
            show_legend=i == 0,
        )
    multi_fig.savefig(plotdir / "dist_combined.pdf", bbox_inches="tight")
    step_multi_fig.savefig(plotdir / "cdf_combined.pdf", bbox_inches="tight")

    # Generate ramachandran plot for sampled angles
    plot_ramachandran(
        final_sampled_stacked[:, phi_idx],
        final_sampled_stacked[:, psi_idx],
        fname=plotdir / "ramachandran_generated.pdf",
    )

    # Generate plots of secondary structure co-occurrence
    make_ss_cooccurrence_plot(
        pdb_files,
        str(outdir / "plots" / "ss_cooccurrence_sampled.pdf"),
        threads=multiprocessing.cpu_count(),
    )
    if args.testcomparison:
        make_ss_cooccurrence_plot(
            test_dset.filenames,
            str(outdir / "plots" / "ss_cooccurrence_test.pdf"),
            max_seq_len=test_dset.dset.pad,
            threads=multiprocessing.cpu_count(),
        )


if __name__ == '__main__':
    main()