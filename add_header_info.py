from pathlib import Path

import pandas as pd
from tqdm import tqdm


from bin.structure_utils import add_sequence_to_pdb_header


def main():
    in_dir_name = "pdb_to_correct/s1"
    out_dir_name = "pdb_to_correct/s1_header"

    # Make the output directory
    Path(out_dir_name).mkdir(parents=True, exist_ok=True)

    csv_file = "pdb_to_correct_debug/seqs_1.csv"
    df = pd.read_csv(csv_file)

    print(df.columns)

    pdb_files = [
        str(f)
        for f in Path(in_dir_name).glob("*.pdb")
    ]

    for pdb_file in tqdm(pdb_files):
        pdb_file_path = Path(pdb_file)
        pdb_file_name = pdb_file_path.stem
        pdb_file_id = pdb_file_name.split("_")[0]
        pdb_file_chain = pdb_file_name.split("_")[1]

        # Take the line in the csv file that matches the pdb file
        pdb_file_df = df.loc[
            (df["pdbid"] == pdb_file_id) &
            (df["interactor_1"] == pdb_file_chain)
        ]

        # Take the sequence
        sequence = pdb_file_df["interactor_1 sequence"].values[0]

        # Compute thew new file_path
        new_file_path = str(Path(out_dir_name) / f"{pdb_file_name}.pdb")

        add_sequence_to_pdb_header(
            in_pdb_file=pdb_file,
            sequence=sequence,
            chain=pdb_file_chain,
            out_pdb_file=new_file_path
        )


if __name__ == "__main__":
    main()