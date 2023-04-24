from pathlib import Path

from biotite.structure import check_bond_continuity
from tqdm import tqdm

from bin.structure_utils import read_pdb_file


def main():
    dir_name = "pdb_corrected_s1_header/fine_tuned"
    pdb_files = [
        str(f)
        for f in Path(dir_name).glob("*.pdb")
    ]

    broken_pdbs = []
    for pdb_file in tqdm(pdb_files):
        structure = read_pdb_file(pdb_file)
        if check_bond_continuity(structure).size != 0:
            broken_pdbs.append(pdb_file)

    print(len(broken_pdbs))
    print(len(pdb_files))

    print(broken_pdbs)

    # s1 - 8
    # s2 - 17


if __name__ == "__main__":
    main()