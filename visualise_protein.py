import gzip

import pandas
from biotite.structure.io.pdb import PDBFile


def read_pdb_file(fname: str):
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        source = PDBFile.read(f)
    if source.get_model_count() > 1:
        return None
    # Pull out the atomarray from atomarraystack
    source_struct = source.get_structure()[0]

    return source_struct


def main():
    fname = "pdb_to_correct/2ZJR_W.pdb"
    # fname = "pdb_corrected/sampled_pdb/generated_1.pdb"
    # fname = "pdb_to_correct/2ZJR_W.pdb"
    # fname = "/home/alexandru/code/foldingdiff/data/cath/dompdb/2pfuA01"
    # fname = "generated_data_now_1/sampled_pdb/generated_0.pdb"
    source_struct = read_pdb_file(fname)
    # filter just the CA
    source_struct = source_struct[source_struct.atom_name == "CA"]

    # Get the atom coordinates
    coords = source_struct.coord

    # Plot the protein in 3d using matplotlib
    from matplotlib import pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Protein")

    color_map = {
        0: 'black',
        16: 'red',
        20: 'green',
    }

    # Set the axes for xyz between -15 and 15
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-15, 15)

    color = "black"
    for i in range(len(coords) - 1):
        # plot lines
        if i in color_map:
            color = color_map[i]

        ax.plot([coords[i][0], coords[i+1][0]], [coords[i][1], coords[i+1][1]], [coords[i][2], coords[i+1][2]], color=color)

    plt.show()

    # make the plot interactive
    plt.ion()
    plt.show()
    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
