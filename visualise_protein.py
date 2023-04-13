from bin.structure_utils import read_pdb_file


def main():
    # fname = "pdb_to_correct/2ZJR_W.pdb"
    # fname = "pdb_corrected/sampled_pdb/generated_1.pdb"
    # fname = "pdb_to_correct/2ZJR_W.pdb"
    # fname = "/home/alexandru/code/foldingdiff/data/cath/dompdb/2pfuA01"
    # fname = "generated_data_now_1/sampled_pdb/generated_0.pdb"

    # fname = "pdb_to_correct/generated_0.pdb"
    # fname = "pdb_corrected/sampled_pdb/generated_0.pdb"
    # fname = "pdb_to_correct/generated_0_long.pdb"

    # fname = "pdb_to_correct/1jrh.pdb"

    # fname = "pdb_to_correct/5f3b.pdb"
    # fname = "pdb_to_correct/6e63.pdb"

    fname = "pdb_to_correct/5f3b.pdb"

    source_struct = read_pdb_file(fname)
    # filter just the CA
    source_struct = source_struct[source_struct.atom_name == "CA"]

    # filter out only for chain C
    source_struct = source_struct[source_struct.chain_id == "C"]

    # only first 30 residues
    source_struct = source_struct[:30]

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
        9: 'red',
        19: 'green',
        130: 'blue',
        140: 'green'
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

        # if source_struct[i].chain_id in color_map:
        #     color = color_map[source_struct[i].chain_id]

        ax.plot(
            [coords[i][0], coords[i+1][0]],
            [coords[i][1], coords[i+1][1]],
            [coords[i][2], coords[i+1][2]],
            color=color
        )

    plt.show()


if __name__ == "__main__":
    main()
