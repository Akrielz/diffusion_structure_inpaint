import gzip

import pandas
from biotite.structure.io.pdb import PDBFile

from foldingdiff.angles_and_coords import canonical_distances_and_dihedrals, EXHAUSTIVE_ANGLES


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
    df = canonical_distances_and_dihedrals(fname, angles=EXHAUSTIVE_ANGLES)
    return df


def main():
    df1 = read_saved_angles()
    df2 = compute_angles_from_pdb("generated_data_now_1/sampled_pdb/generated_0.pdb")

    print("hi")


if __name__ == "__main__":
    main()