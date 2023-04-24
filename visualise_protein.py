import biotite

from bin.structure_utils import read_pdb_file


def main():
    # fname = "pdb_to_correct/s1/1AHW_C.pdb"
    # fname = "pdb_to_correct/2ZJR_W_broken.pdb"
    # fname = "pdb_to_correct/mocked.pdb"
    # fname = "pdb_corrected/best_pdb/original_best.pdb"
    # fname = "pdb_corrected/fine_tuned/1AHW_C.pdb"
    # fname = "pdb_corrected/corrected_structures/1AHW_C.pdb"
    fname = "pdb_corrected_s1_header/fine_tuned/4K24_U.pdb"

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
        0: 'green',
        82: 'red',
        90: 'green',
    }

    # limit = 50
    # ax.set_xlim(-limit, limit)
    # ax.set_ylim(-limit, limit)
    # ax.set_zlim(-limit, limit)

    color = "black"
    for i in range(len(coords) - 1):
        # # plot lines
        if i in color_map:
            color = color_map[i]

        ax.plot(
            [coords[i][0], coords[i+1][0]],
            [coords[i][1], coords[i+1][1]],
            [coords[i][2], coords[i+1][2]],
            color=color
        )

    plt.show()


if __name__ == "__main__":
    main()
