"""
Code for sampling from diffusion models
"""
import json
import os
import multiprocessing as mp
from pathlib import Path
import tempfile
import logging
from typing import *

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import default_collate
from huggingface_hub import snapshot_download

from foldingdiff import datasets as dsets
from foldingdiff import beta_schedules, modelling, utils, sampling, tmalign
from foldingdiff import angles_and_coords as ac


@torch.no_grad()
def p_sample(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    seq_lens: Sequence[int],
    t_index: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """
    Sample the given timestep. Note that this _may_ fall off the manifold if we just
    feed the output back into itself repeatedly, so we need to perform modulo on it
    (see p_sample_loop)
    """
    # Calculate alphas and betas
    alpha_beta_values = beta_schedules.compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"])

    # Select based on time
    t_unique = torch.unique(t)
    assert len(t_unique) == 1, f"Got multiple values for t: {t_unique}"
    t_index = t_unique.item()
    sqrt_recip_alphas_t = sqrt_recip_alphas[t_index]
    betas_t = betas[t_index]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values[
        "sqrt_one_minus_alphas_cumprod"
    ][t_index]

    # Create the attention mask
    attn_mask = torch.zeros(x.shape[:2], device=x.device)
    for i, l in enumerate(seq_lens):
        attn_mask[i, :l] = 1.0

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_result = model(x, t, attention_mask=attn_mask)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model_result / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop_missing_info(
        model: nn.Module,
        lengths: Sequence[int],
        noise_list: torch.Tensor,
        original_angles: torch.Tensor,
        missing_info_mask: torch.Tensor,
        timesteps: int,
        betas: torch.Tensor,
        is_angle: List[bool],
        disable_pbar: bool = False,
) -> torch.Tensor:
    """
    Returns a tensor of shape (timesteps, batch_size, seq_len, n_ft)
    """

    # Get device
    device = next(model.parameters()).device

    # Get batch size
    b = noise_list[0].shape[0]

    # Compute the mask for the missing info
    not_missing_info_mask = 1 - missing_info_mask.long()
    not_missing_info_mask = not_missing_info_mask.to(torch.bool)
    not_missing_info_mask = not_missing_info_mask.to(device)
    not_missing_info_mask = not_missing_info_mask.repeat(b, 1)

    # Put the noise list on the device
    noise_list = noise_list.to(device)
    full_noise = noise_list[-1]

    original_angles = original_angles.to(device)

    img = full_noise.to(device)

    # Report metrics on starting noise
    # amin and amax support reducing on multiple dimensions
    logging.info(
        f"Starting from noise {full_noise.shape} with angularity {is_angle} "
        f"and range {torch.amin(img, dim=(0, 1))} - {torch.amax(img, dim=(0, 1))} using {device}"
    )

    imgs = []

    for t in tqdm(
        reversed(range(0, timesteps)),
        desc="sampling loop time step",
        total=timesteps,
        disable=disable_pbar,
    ):
        # Replace the non-missing info with the adequate angles noise
        img[not_missing_info_mask] = noise_list[t][not_missing_info_mask]

        # Shape is (batch, seq_len, 6)
        img = p_sample(
            model=model,
            x=img,
            t=torch.full((b,), t, device=device, dtype=torch.long),  # time vector
            seq_lens=lengths,
            t_index=t,
            betas=betas,
        )

        # Wrap if angular
        if isinstance(is_angle, bool):
            if is_angle:
                img = utils.modulo_with_wrapped_range(
                    img, range_min=-torch.pi, range_max=torch.pi
                )
        else:
            assert len(is_angle) == img.shape[-1]
            for j in range(img.shape[-1]):
                if is_angle[j]:
                    img[:, :, j] = utils.modulo_with_wrapped_range(
                        img[:, :, j], range_min=-torch.pi, range_max=torch.pi
                    )

        imgs.append(img.cpu())

    imgs[-1][not_missing_info_mask] = original_angles.cpu()[not_missing_info_mask]
    return torch.stack(imgs)


@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    lengths: Sequence[int],
    noise: torch.Tensor,
    timesteps: int,
    betas: torch.Tensor,
    is_angle: Union[bool, List[bool]] = [False, True, True, True],
    disable_pbar: bool = False,
) -> torch.Tensor:
    """
    Returns a tensor of shape (timesteps, batch_size, seq_len, n_ft)
    """
    device = next(model.parameters()).device
    b = noise.shape[0]
    img = noise.to(device)
    # Report metrics on starting noise
    # amin and amax support reducing on multiple dimensions
    logging.info(
        f"Starting from noise {noise.shape} with angularity {is_angle} and range {torch.amin(img, dim=(0, 1))} - {torch.amax(img, dim=(0, 1))} using {device}"
    )

    imgs = []

    for i in tqdm(
        reversed(range(0, timesteps)),
        desc="sampling loop time step",
        total=timesteps,
        disable=disable_pbar,
    ):
        # Shape is (batch, seq_len, 4)
        img = p_sample(
            model=model,
            x=img,
            t=torch.full((b,), i, device=device, dtype=torch.long),  # time vector
            seq_lens=lengths,
            t_index=i,
            betas=betas,
        )

        # Wrap if angular
        if isinstance(is_angle, bool):
            if is_angle:
                img = utils.modulo_with_wrapped_range(
                    img, range_min=-torch.pi, range_max=torch.pi
                )
        else:
            assert len(is_angle) == img.shape[-1]
            for j in range(img.shape[-1]):
                if is_angle[j]:
                    img[:, :, j] = utils.modulo_with_wrapped_range(
                        img[:, :, j], range_min=-torch.pi, range_max=torch.pi
                    )
        imgs.append(img.cpu())
    return torch.stack(imgs)


def sample_missing_structure(
        model: nn.Module,
        train_dset: dsets.NoisedAnglesDataset,
        original_length: int,
        missing_info_mask: torch.Tensor,
        original_features: Dict[str, torch.Tensor],
        n: int = 10,
        batch_size: int = 512,
        feature_key: str = "angles",
        disable_pbar: bool = False,
        window_step: int = 32,
        window_size: int = 128,
        whole_pad_len: int = 128,
):
    # Convert missing info mask to bool tensor
    missing_info_mask = missing_info_mask.to(torch.bool)

    # Get original angles
    original_angles: torch.Tensor = original_features["angles"]
    infilled_angles = original_angles.clone().repeat(n, 1, 1)

    # Compute the left index of the window
    left_indexes = torch.arange(0, whole_pad_len - window_size + 1, window_step)

    for left in left_indexes:
        right = left + window_size
        cropped_attn_mask = original_features['attn_mask'][:, left:right]
        cropped_missing_info_mask = missing_info_mask[:, left:right]

        # If there is nothing to infill, skip
        if cropped_missing_info_mask.sum() == 0:
            continue

        # If the infilled part is not in the first half too, move the window to give it more context
        if left != left_indexes[-1] and cropped_missing_info_mask[:, :window_size // 2].sum() == 0:
            continue

        cropped_angles = infilled_angles[:, left:right]
        cropped_real_len = cropped_attn_mask.sum()

        # Compute lengths for each sample
        lengths = [cropped_real_len.long().item()] * n
        lengths_batches = [
            lengths[i: i + batch_size] for i in range(0, len(lengths), batch_size)
        ]

        cropped_angles_batches = [
            cropped_angles[i: i + batch_size] for i in range(0, len(cropped_angles), batch_size)
        ]

        chunk_values = []
        for cropped_angles_batch, lengths_batch in zip(cropped_angles_batches, lengths_batches):
            noise_list = train_dset.sample_full_list_of_noise(cropped_angles_batch)

            sampled_angles = p_sample_loop_missing_info(
                model=model,
                lengths=lengths_batch,
                noise_list=noise_list,
                timesteps=train_dset.timesteps,
                betas=train_dset.alpha_beta_terms["betas"],
                is_angle=train_dset.feature_is_angular[feature_key],
                disable_pbar=disable_pbar,
                original_angles=cropped_angles_batch,
                missing_info_mask=cropped_missing_info_mask
            )
            chunk_values.extend(sampled_angles[-1])

        # Get all the sampled angles together and replace info in infilled_angles
        sampled_angles_chunk = torch.stack(chunk_values)
        infilled_angles[:, left:right] = sampled_angles_chunk
        missing_info_mask[:, left:right] = False

    # Trim the infilled angles to the original length
    trimmed_sampled = [
        infilled_angles[i, :original_length, :].numpy() for i in range(n)
    ]

    return_values = mean_shift(feature_key, trimmed_sampled, train_dset)

    return return_values


def mean_shift(feature_key, return_values, train_dset):
    if (
            hasattr(train_dset, "dset")
            and hasattr(train_dset.dset, "get_masked_means")
            and train_dset.dset.get_masked_means() is not None
    ):
        logging.info(
            f"Shifting predicted values by original offset: {train_dset.dset.get_masked_means()}"
        )
        return_values = [s + train_dset.dset.get_masked_means() for s in return_values]
        # Because shifting may have caused us to go across the circle boundary, re-wrap
        angular_idx = np.where(train_dset.feature_is_angular[feature_key])[0]
        for s in return_values:
            s[..., angular_idx] = utils.modulo_with_wrapped_range(
                s[..., angular_idx], range_min=-np.pi, range_max=np.pi
            )
    return return_values


def sample(
    model: nn.Module,
    train_dset: dsets.NoisedAnglesDataset,
    n: int = 10,
    sweep_lengths: Optional[Tuple[int, int]] = (50, 128),
    batch_size: int = 512,
    feature_key: str = "angles",
    disable_pbar: bool = False,
) -> List[np.ndarray]:
    """
    Sample from the given model. Use the train_dset to generate noise to sample
    sequence lengths. Returns a list of arrays, shape (timesteps, seq_len, fts).
    If sweep_lengths is set, we generate n items per length in the sweep range

    train_dset object must support:
    - sample_noise - provided by NoisedAnglesDataset
    - timesteps - provided by NoisedAnglesDataset
    - alpha_beta_terms - provided by NoisedAnglesDataset
    - feature_is_angular - provided by *wrapped dataset* under NoisedAnglesDataset
    - pad - provided by *wrapped dataset* under NoisedAnglesDataset
    And optionally, sample_length()
    """
    # Process each batch
    if sweep_lengths is not None:
        sweep_min, sweep_max = sweep_lengths
        logging.info(
            f"Sweeping from {sweep_min}-{sweep_max} with {n} examples at each length"
        )
        lengths = []
        for l in range(sweep_min, sweep_max):
            lengths.extend([l] * n)
    else:
        lengths = [train_dset.sample_length() for _ in range(n)]
    lengths_chunkified = [
        lengths[i : i + batch_size] for i in range(0, len(lengths), batch_size)
    ]

    logging.info(f"Sampling {len(lengths)} items in batches of size {batch_size}")
    retval = []
    for this_lengths in lengths_chunkified:
        batch = len(this_lengths)
        # Sample noise and sample the lengths
        noise = train_dset.sample_noise(
            torch.zeros((batch, train_dset.pad, model.n_inputs), dtype=torch.float32)
        )
        # Produces (timesteps, batch_size, seq_len, n_ft)
        sampled = p_sample_loop(
            model=model,
            lengths=this_lengths,
            noise=noise,
            timesteps=train_dset.timesteps,
            betas=train_dset.alpha_beta_terms["betas"],
            is_angle=train_dset.feature_is_angular[feature_key],
            disable_pbar=disable_pbar,
        )
        # Gets to size (timesteps, seq_len, n_ft)
        trimmed_sampled = [
            sampled[:, i, :l, :].numpy() for i, l in enumerate(this_lengths)
        ]
        retval.extend(trimmed_sampled)
    # Note that we don't use means variable here directly because we may need a subset
    # of it based on which features are active in the dataset. The function
    # get_masked_means handles this gracefully
    if (
        hasattr(train_dset, "dset")
        and hasattr(train_dset.dset, "get_masked_means")
        and train_dset.dset.get_masked_means() is not None
    ):
        logging.info(
            f"Shifting predicted values by original offset: {train_dset.dset.get_masked_means()}"
        )
        retval = [s + train_dset.dset.get_masked_means() for s in retval]
        # Because shifting may have caused us to go across the circle boundary, re-wrap
        angular_idx = np.where(train_dset.feature_is_angular[feature_key])[0]
        for s in retval:
            s[..., angular_idx] = utils.modulo_with_wrapped_range(
                s[..., angular_idx], range_min=-np.pi, range_max=np.pi
            )

    return retval


def sample_simple(
    model_dir: str, n: int = 10, sweep_lengths: Tuple[int, int] = (50, 128)
) -> List[pd.DataFrame]:
    """
    Simple wrapper on sample to automatically load in the model and dummy dataset
    Primarily for gradio integration
    """
    if utils.is_huggingface_hub_id(model_dir):
        model_dir = snapshot_download(model_dir)
    assert os.path.isdir(model_dir)

    with open(os.path.join(model_dir, "training_args.json")) as source:
        training_args = json.load(source)

    model = modelling.BertForDiffusionBase.from_dir(model_dir)
    if torch.cuda.is_available():
        model = model.to("cuda:0")

    dummy_dset = dsets.AnglesEmptyDataset.from_dir(model_dir)
    dummy_noised_dset = dsets.NoisedAnglesDataset(
        dset=dummy_dset,
        dset_key="coords" if training_args == "cart-cords" else "angles",
        timesteps=training_args["timesteps"],
        exhaustive_t=False,
        beta_schedule=training_args["variance_schedule"],
        nonangular_variance=1.0,
        angular_variance=training_args["variance_scale"],
    )

    sampled = sample(
        model, dummy_noised_dset, n=n, sweep_lengths=sweep_lengths, disable_pbar=True
    )
    final_sampled = [s[-1] for s in sampled]
    sampled_dfs = [
        pd.DataFrame(s, columns=dummy_noised_dset.feature_names["angles"])
        for s in final_sampled
    ]
    return sampled_dfs


def _score_angles(
    reconst_angles:pd.DataFrame, truth_angles:pd.DataFrame, truth_coords_pdb: str
) -> Tuple[float, float]:
    """
    Helper function to scores sets of angles
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        truth_path = Path(tmpdir) / "truth.pdb"
        reconst_path = Path(tmpdir) / "reconst.pdb"

        truth_pdb = ac.create_new_chain_nerf_to_file(str(truth_path), truth_angles)
        reconst_pdb = ac.create_new_chain_nerf_to_file(str(reconst_path), reconst_angles)

        # Calculate WRT the truth angles
        score = tmalign.run_tmalign(reconst_pdb, truth_pdb)

        score_coord = tmalign.run_tmalign(reconst_pdb, truth_coords_pdb)
    return score, score_coord


@torch.no_grad()
def get_reconstruction_error(
    model: nn.Module, dset, noise_timesteps: int = 250, bs: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the reconstruction error when adding <noise_timesteps> noise to the idx-th
    item in the dataset.
    """
    device = next(model.parameters()).device
    model.eval()

    recont_angle_sets = []
    truth_angle_sets = []
    truth_pdb_files = []
    for idx_batch in tqdm(utils.seq_to_groups(list(range(len(dset))), bs)):
        batch = default_collate(
            [
                {
                    k: v.to(device)
                    for k, v in dset.__getitem__(idx, use_t_val=noise_timesteps).items()
                }
                for idx in idx_batch
            ]
        )
        img = batch["corrupted"].clone()
        assert img.ndim == 3

        # Record the actual files containing raw coordinates
        for i in idx_batch:
            truth_pdb_files.append(dset.filenames[i])

        # Run the diffusion model for noise_timesteps steps
        for i in tqdm(list(reversed(list(range(0, noise_timesteps))))):
            img = sampling.p_sample(
                model=model,
                x=img,
                t=torch.full((len(idx_batch),), fill_value=i, dtype=torch.long).to(
                    device
                ),
                seq_lens=batch["lengths"],
                t_index=i,
                betas=dset.alpha_beta_terms["betas"],
            )
            img = utils.modulo_with_wrapped_range(img)

        # Finished reconstruction, subset to lengths and add to running list
        for i, l in enumerate(batch["lengths"].squeeze()):
            recont_angle_sets.append(
                pd.DataFrame(img[i, :l].cpu().numpy(), columns=ac.EXHAUSTIVE_ANGLES)
            )
            truth_angle_sets.append(
                pd.DataFrame(
                    batch["angles"][i, :l].cpu().numpy(), columns=ac.EXHAUSTIVE_ANGLES
                )
            )

    # Get the reconstruction error as a TM score
    logging.info(
        f"Calculating TM scores for reconstruction error with {mp.cpu_count()} processes"
    )
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.starmap(
        _score_angles,
        zip(recont_angle_sets, truth_angle_sets, truth_pdb_files),
        chunksize=10,
    )
    pool.close()
    pool.join()
    scores, coord_scores = zip(*results)
    return np.array(scores), np.array(coord_scores)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = sample_simple("wukevin/foldingdiff_cath", n=1, sweep_lengths=(50, 55))
    for i, x in enumerate(s):
        print(x.shape)
