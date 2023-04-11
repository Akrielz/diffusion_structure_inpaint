import gzip
from typing import Dict

import pandas
import torch
from biotite.structure.io.pdb import PDBFile
from torch.utils.data import DataLoader

from foldingdiff.angles_and_coords import canonical_distances_and_dihedrals, EXHAUSTIVE_ANGLES, EXHAUSTIVE_DISTS
from foldingdiff.datasets import CathCanonicalAnglesOnlyDataset, NoisedAnglesDataset


def read_pdb_file(fname: str):
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        source = PDBFile.read(f)
    if source.get_model_count() > 1:
        return None
    # Pull out the atomarray from atomarraystack
    source_struct = source.get_structure()[0]

    return source_struct


def read_saved_angles():
    # read generated_data_now_1/sampled_angles/generated_0.csv.gz
    df = pandas.read_csv("generated_data_now_1/sampled_angles/generated_0.csv.gz")

    return df


def compute_angles_from_pdb(fname: str):
    df = canonical_distances_and_dihedrals(fname, angles=EXHAUSTIVE_ANGLES, distances=EXHAUSTIVE_DISTS)
    return df


def read_to_correct_structure(pdb_file: str, pad_len=128) -> Dict[str, torch.Tensor]:
    clean_dset = CathCanonicalAnglesOnlyDataset(
        pad=pad_len,
        trim_strategy='',
        fnames=[pdb_file],
        use_cache=False,
    )
    # noised_dset = NoisedAnglesDataset(
    #     clean_dset,
    #     timesteps=1000,
    #     beta_schedule='cosine'
    # )
    dl = DataLoader(clean_dset, batch_size=32, shuffle=False)
    features = iter(dl).next()

    return features['angles']


def main():
    df1 = read_saved_angles()
    df2 = compute_angles_from_pdb("generated_data_now_1/sampled_pdb/generated_0.pdb")
    their_angles = read_to_correct_structure("generated_data_now_1/sampled_pdb/generated_0.pdb")

    print("hi")


if __name__ == "__main__":
    main()