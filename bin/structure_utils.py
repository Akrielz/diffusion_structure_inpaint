import gzip
import json
from collections import defaultdict
from copy import deepcopy
from typing import Optional, List, Dict, Any

import biotite
import numpy as np
from Bio import Align
from Bio.pairwise2 import Alignment
from biotite.structure import AtomArray, filter_backbone, Atom
from biotite.structure import array as struct_array
from biotite.structure.io.pdb import PDBFile
from matplotlib import pyplot as plt

from foldingdiff.angles_and_coords import write_atom_array_to_pdb

d3to1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}

d1to3 = {v: k for k, v in d3to1.items()}


MissingResidues = List[int]


def order_keys_in_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: d[k]
        for k in sorted(d.keys())
    }


def get_pdb_file_pointer(file_name) -> PDBFile:
    opener = gzip.open if file_name.endswith(".gz") else open
    with opener(str(file_name), "rt") as f:
        source = PDBFile.read(f)

    return source


def read_pdb_file(file_name: str) -> Optional[AtomArray]:
    source = get_pdb_file_pointer(file_name)
    if source.get_model_count() > 1:
        return None
    # Pull out the atomarray from atomarraystack
    source_struct = source.get_structure()[0]

    return source_struct


def filter_only_ca(structure: AtomArray) -> AtomArray:
    return structure[structure.atom_name == "CA"]


def get_chains(structure: AtomArray) -> List[str]:
    return list(set(structure.chain_id))


def split_structure_by_chains(structure: AtomArray) -> Dict[str, AtomArray]:
    chains = get_chains(structure)
    return {
        chain: structure[structure.chain_id == chain]
        for chain in chains
    }


def get_num_residues(structure: AtomArray) -> int:
    return max(set(structure.res_id))


def get_lowest_residue(structure: AtomArray) -> int:
    # In case of missing residues, the lowest residue is not 1
    # Also in case there are residues indexed with negative numbers
    # or with 0
    return min(1, min(set(structure.res_id)))


def iter_residues(structure: AtomArray) -> range:
    return range(get_lowest_residue(structure), get_num_residues(structure) + 1)


def get_num_residues_by_chains(structure: AtomArray) -> Dict[str, int]:
    chains = get_chains(structure)
    return {
        chain: get_num_residues(structure[structure.chain_id == chain])
        for chain in chains
    }


def get_num_residues_backbone_by_chains(structure: AtomArray) -> Dict[str, int]:
    chains = get_chains(structure)
    structure = filter_only_ca(structure)
    return {
        chain: get_num_residues(structure[structure.chain_id == chain])
        for chain in chains
    }


def read_pdb_header(file_name: str) -> List[str]:
    source = get_pdb_file_pointer(file_name)

    # get all the lines until the first ATOM

    body_lines = ["ATOM", "HETATM"]

    stop_pointer = 0
    for i, line in enumerate(source.lines):
        for body_line in body_lines:
            if line.startswith(body_line):
                stop_pointer = i
                break

    return source.lines[:stop_pointer]


def get_sequence_from_header(header: List[str]) -> Optional[Dict[str, str]]:
    seqres_lines = [
        line for line in header
        if line.startswith("SEQRES")
    ]

    if len(seqres_lines) == 0:
        return None

    # A seqres example
    # 'SEQRES   4 D  109  ALA ASN TYR CYS SER GLY GLU CYS GLU PHE VAL PHE LEU          '

    # Split the lines in columns
    seq_res_columns = [line.split() for line in seqres_lines]

    # Get the sequence from each line
    current_chain = None
    amino_acids_by_chain = defaultdict(list)
    for line in seq_res_columns:
        for i, column in enumerate(line):
            if i == 0 or i == 1 or i == 3:
                continue

            if i == 2:
                current_chain = column
                continue

            amino_acid = d3to1.get(column, "?")
            amino_acids_by_chain[current_chain].append(amino_acid)

    # Join the sequences
    amino_acids_by_chain = {
        chain: "".join(seq)
        for chain, seq in amino_acids_by_chain.items()
    }

    return amino_acids_by_chain


def get_sequence_from_structure(structure: AtomArray) -> Dict[str, str]:
    chains = get_chains(structure)
    structure = filter_only_ca(structure)
    return {
        chain: "".join(
            d3to1.get(res_name, "?")
            for res_name in structure[structure.chain_id == chain].res_name
        )
        for chain in chains
    }


def sequence_alignments(header_seq: str, structure_seq: str) -> Alignment:
    aligner = Align.PairwiseAligner()
    aligner.gap_score = -1
    aligner.mismatch_score = -1
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(header_seq, structure_seq)

    return alignments


def missing_residues_by_sequence_alignment(
        header_seq: str,
        structure_seq: str,
        all_alignments: bool = False
) -> List[MissingResidues]:

    alignments = sequence_alignments(header_seq, structure_seq)

    missing_residues_by_alignment = []
    for alignment in alignments:
        missing_residues = missing_residue_in_alignment(alignment)
        missing_residues_by_alignment.append(missing_residues)

        if not all_alignments:
            break

    return missing_residues_by_alignment


def missing_residue_in_alignment(alignment: Alignment) -> MissingResidues:
    missing_residues = []
    for i, (header_res, structure_res) in enumerate(zip(alignment[0], alignment[1])):
        if header_res != structure_res:
            missing_residues.append(i + 1)

    return missing_residues


def missing_residues_by_sequence_alignment_by_chains(
        header_seq: Dict[str, str],
        structure_seq: Dict[str, str],
        all_alignments: bool = False
) -> Dict[str, List[MissingResidues]]:
    missing_residues_by_chains = {}
    for chain in header_seq.keys():
        missing_residues_by_chains[chain] = missing_residues_by_sequence_alignment(
            header_seq[chain], structure_seq[chain], all_alignments=all_alignments
        )

    return missing_residues_by_chains


def missing_residues_by_structure_look_residues_id(structure: AtomArray) -> Dict[str, MissingResidues]:
    chains = get_chains(structure)
    structure = filter_only_ca(structure)
    missing_residues_by_chains = {}
    for chain in chains:
        missing_residues_by_chains[chain] = []
        chain_structure = structure[structure.chain_id == chain]

        for i in iter_residues(chain_structure):
            if len(chain_structure[chain_structure.res_id == i]) == 0:
                missing_residues_by_chains[chain].append(i)

    return missing_residues_by_chains


def missing_residues_by_structure_continuity(structure: AtomArray) -> Dict[str, MissingResidues]:
    chains = get_chains(structure)
    missing_residues_by_chains = {}
    for chain in chains:
        chain_structure = structure[(structure.chain_id == chain)]
        stop_indexes_atoms: np.ndarray = biotite.structure.check_bond_continuity(chain_structure)

        start_indexes_residues = chain_structure[stop_indexes_atoms-1].res_id
        stop_indexes_residues = chain_structure[stop_indexes_atoms].res_id

        missing_residues_by_chains[chain] = []
        for start, stop in zip(start_indexes_residues, stop_indexes_residues):
            missing_residues_by_chains[chain].extend(list(range(start+1, stop)))

        missing_residues_by_chains[chain] = sorted(list(set(missing_residues_by_chains[chain])))

    return missing_residues_by_chains


def broken_residues_by_structure(structure: AtomArray):
    backbone_mask = filter_backbone(structure)
    backbone_structure = structure[backbone_mask]
    chains = get_chains(structure)

    broken_residues_by_chains = {}
    for chain in chains:
        broken_residues_by_chains[chain] = []

        chain_structure = backbone_structure[backbone_structure.chain_id == chain]

        for i in iter_residues(chain_structure):
            backbone_residue = chain_structure[chain_structure.res_id == i]
            n_atom = backbone_residue[backbone_residue.atom_name == "N"]
            ca_atom = backbone_residue[backbone_residue.atom_name == "CA"]
            c_atom = backbone_residue[backbone_residue.atom_name == "C"]

            if len(backbone_residue) == 0:
                continue

            if len(n_atom) == 0 or len(ca_atom) == 0 or len(c_atom) == 0:
                broken_residues_by_chains[chain].append(i)

    return broken_residues_by_chains


def determine_missing_residues(pdb_file: str) -> Dict[str, MissingResidues]:

    # Get the info
    header = read_pdb_header(pdb_file)
    structure = read_pdb_file(pdb_file)

    # Compute all the missing residues possibilities
    header_seq = get_sequence_from_header(header)
    missing_residues_header = None
    if header_seq is not None:
        # If the sequence is not in the header, we can't compute the missing residues with this method
        structure_seq = get_sequence_from_structure(structure)
        missing_residues_header = missing_residues_by_sequence_alignment_by_chains(
            header_seq, structure_seq, all_alignments=True
        )

    missing_residues_struct_res = missing_residues_by_structure_look_residues_id(structure)
    missing_residues_struct_cont = missing_residues_by_structure_continuity(structure)
    broken_residues_struct = broken_residues_by_structure(structure)

    # Check if the missing residues correspond
    for chain in missing_residues_struct_res.keys():
        if set(missing_residues_struct_cont[chain]).issubset(set(missing_residues_struct_res[chain])):
            continue

        # warning
        print("Warning: Different subsets of missing residues found with different methods for chain", chain)
        print("Missing residues by structure continuity:", missing_residues_struct_cont[chain])
        print("Missing residues by structure residues id:", missing_residues_struct_res[chain])

    missing_residues = missing_residues_struct_res
    if missing_residues_header is not None:

        # Reindex the missing residues by header
        structure = reindex_alignments(missing_residues_header, structure)

        # We need to drop all the alignments which include already existent residues
        missing_residues_header = keep_only_plausible_alignments(missing_residues_header, structure)

        # We need to check all the given alignments to decide which is the best for each chain
        for chain in missing_residues_header.keys():
            alignment_found = False
            for alignment in missing_residues_header[chain]:
                if alignment_found:
                    break

                if set(missing_residues_struct_res[chain]).issubset(set(alignment)):
                    continue

                # If the alignment is correct, we can use it
                missing_residues[chain] = alignment
                alignment_found = True

            # If we didn't find any alignment, we will use the first alignment from the header
            # which also has the highest score
            if not alignment_found and len(missing_residues_header[chain]) > 0:
                missing_residues[chain] = missing_residues_header[chain][0]

    # And now we augment the missing residues with the broken residues
    for chain in broken_residues_struct.keys():
        missing_residues[chain].extend(broken_residues_struct[chain])
        missing_residues[chain] = sorted(list(set(missing_residues[chain])))

    return missing_residues


def reindex_alignments(
        missing_residues_header: Dict[str, List[MissingResidues]],
        structure: AtomArray
):
    for chain in missing_residues_header.keys():
        chain_struct = structure[structure.chain_id == chain]
        start_index = get_lowest_residue(chain_struct)

        for i, alignment in enumerate(missing_residues_header[chain]):
            indexes = np.array(alignment) + start_index - 1
            missing_residues_header[chain][i] = indexes.tolist()

    return structure


def keep_only_plausible_alignments(missing_residues_header, structure):
    missing_residues_plausible = defaultdict(list)
    for chain in missing_residues_header.keys():
        for i, alignment in enumerate(missing_residues_header[chain]):
            # check if all the residues in alignment exist in the structure
            structure_chain = structure[structure.chain_id == chain]
            res_ids_in_structure = set(structure_chain.res_id)

            # Check if any of the residues in the alignment is in the structure
            if len(set(alignment).intersection(res_ids_in_structure)) > 0:
                continue

            # If not, we can use this alignment
            missing_residues_plausible[chain].append(alignment)
    missing_residues_header = missing_residues_plausible
    return missing_residues_header


def get_all_residues_id(
        structure: AtomArray,
        missing_residues_id: Dict[str, MissingResidues]
) -> Dict[str, List[int]]:

    all_residues_id = {
        chain: []
        for chain in missing_residues_id.keys()
    }
    for chain in missing_residues_id.keys():
        chain_struct = structure[structure.chain_id == chain]
        chain_residues = set(missing_residues_id[chain]).union(set(chain_struct.res_id))
        all_residues_id[chain] = sorted(list(chain_residues))

    return all_residues_id


def mock_missing_info(
        in_pdb_file: str,
        out_pdb_file: str
):
    structure = read_pdb_file(in_pdb_file)
    missing_residues_id = determine_missing_residues(in_pdb_file)
    all_residues_id = get_all_residues_id(structure, missing_residues_id)

    header = read_pdb_header(in_pdb_file)
    header_seq = get_sequence_from_header(header)
    chains = get_chains(structure)

    new_residues = {
        chain: []
        for chain in chains
    }

    for chain in chains:
        chain_structure = structure[structure.chain_id == chain]
        start_index = get_lowest_residue(chain_structure)

        for residue_id in all_residues_id[chain]:

            if residue_id not in missing_residues_id[chain]:
                residue = chain_structure[chain_structure.res_id == residue_id]
                new_residues[chain].append(residue)
                continue

            missing_aa_3 = "GLY"
            # This is the case for broken existent residues
            if residue_id in chain_structure.res_id:
                first_atom = chain_structure[chain_structure.res_id == residue_id][0]
                missing_aa_3 = first_atom.res_name

            # This is the case for missing residues
            elif header_seq is not None:
                index = residue_id - start_index
                missing_aa = header_seq[chain][index]
                missing_aa_3 = d1to3[missing_aa]

            n_atom = Atom(
                coord=np.random.rand(3),
                chain_id=chain,
                res_id=residue_id,
                res_name=missing_aa_3,
                atom_name="N",
                element="N",
            )

            ca_atom = Atom(
                coord=np.random.rand(3),
                chain_id=chain,
                res_id=residue_id,
                res_name=missing_aa_3,
                atom_name="CA",
                element="C",
            )

            c_atom = Atom(
                coord=np.random.rand(3),
                chain_id=chain,
                res_id=residue_id,
                res_name=missing_aa_3,
                atom_name="C",
                element="C",
            )

            residue = [n_atom, ca_atom, c_atom]
            new_residue = struct_array(residue)
            new_residues[chain].append(new_residue)

    atom_arrays = []
    for chain in chains:
        for residue in new_residues[chain]:
            atom_arrays.extend(residue)

    new_atom_array = struct_array(atom_arrays)
    write_atom_array_to_pdb(new_atom_array, out_pdb_file)

    # add the old header to the new file
    add_header_info(header, out_pdb_file)

    start_indexes = {
        chain: get_lowest_residue(structure[structure.chain_id == chain])
        for chain in chains
    }

    missing_info = {
        "missing_residues_id": missing_residues_id,
        "start_indexes": start_indexes
    }

    # write in out_pdb_file + ".missing" the missing residues as a json
    with open(out_pdb_file + ".missing", "w") as f:
        json.dump(missing_info, f)

    return new_atom_array


def add_header_info(
        header: List[str],
        out_pdb_file: str
):
    with open(out_pdb_file, "r") as f:
        atom_lines = f.readlines()

    new_pdb_lines = deepcopy(header)
    new_pdb_lines[-1] += "\n"
    new_pdb_lines.extend(atom_lines)

    with open(out_pdb_file, "w") as f:
        f.write("".join(new_pdb_lines))


def determine_quality_of_structure(
        structure: AtomArray,
):
    """
    This is a very simple function that determines the quality of a structure based
    on the distances between the backbone atoms. Because of that, it only works
    for fully defined structures.

    The lower the score, the better the structure.

    Quality range: [0, inf)
    """

    n_ca_dist = 1.46  # Check, approximately right
    ca_c_dist = 1.54  # Check, approximately right
    c_n_dist = 1.34  # Check, approximately right

    chains = get_chains(structure)
    quality = {
        chain: 0.0
        for chain in chains
    }

    for chain in chains:
        chain_structure = structure[structure.chain_id == chain]

        # filter just the backbone of the chain
        backbone = chain_structure[
            filter_backbone(chain_structure)
        ]

        # get the coordinates of the backbone
        backbone_coords = backbone.coord

        # get the distances between the backbone atoms
        atom_dist = np.linalg.norm(backbone_coords[1:] - backbone_coords[:-1], axis=1)

        n_ca_len_diff = np.abs(atom_dist[::3] - n_ca_dist)
        ca_c_len_diff = np.abs(atom_dist[1::3] - ca_c_dist)
        c_n_len_diff = np.abs(atom_dist[2::3] - c_n_dist)

        # the quality of a chain is the sum of the differences
        quality[chain] = np.sum(
            np.concatenate([n_ca_len_diff, ca_c_len_diff, c_n_len_diff])
        )

    # Quality score is the mean of the quality of each chain
    quality_score = np.mean(list(quality.values()))

    return quality_score


def compute_backbone_distances(structure: AtomArray):
    chains = get_chains(structure)

    backbone_distances = {
        chain: {
            "n_ca": [],
            "ca_c": [],
            "c_n": [],
        }
        for chain in chains
    }
    for chain in chains:
        chain_structure = structure[structure.chain_id == chain]

        # filter just the backbone of the chain
        backbone = chain_structure[
            filter_backbone(chain_structure)
        ]

        # get the coordinates of the backbone
        backbone_coords = backbone.coord

        # get the distances between the backbone atoms
        atom_dist = np.linalg.norm(backbone_coords[1:] - backbone_coords[:-1], axis=1)

        n_ca_dist = atom_dist[::3]
        ca_c_dist = atom_dist[1::3]
        c_n_dist = atom_dist[2::3]

        backbone_distances[chain]["n_ca"] = n_ca_dist
        backbone_distances[chain]["ca_c"] = ca_c_dist
        backbone_distances[chain]["c_n"] = c_n_dist

    return backbone_distances


def compute_median_backbone_distances(structure: AtomArray):
    backbone_distances = compute_backbone_distances(structure)

    median_backbone_distances = {
        "n_ca": np.median(np.concatenate([backbone_distances[chain]["n_ca"] for chain in backbone_distances])),
        "ca_c": np.median(np.concatenate([backbone_distances[chain]["ca_c"] for chain in backbone_distances])),
        "c_n": np.median(np.concatenate([backbone_distances[chain]["c_n"] for chain in backbone_distances])),
    }

    return median_backbone_distances


def main():
    # file_name = "pdb_to_correct/2ZJR_W.pdb"
    # file_name = "pdb_to_correct/5f3b.pdb"
    # file_name = "pdb_to_correct/2ZJR_W_broken.pdb"
    # file_name = "pdb_to_correct/6fp7.pdb"

    file_names = [
        f"pdb_corrected/sampled_pdb/generated_{i}.pdb"
        for i in range(128)
    ]

    qualities = []
    for file_name in file_names:
        structure = read_pdb_file(file_name)
        quality = determine_quality_of_structure(structure)
        qualities.append(quality)

    print(qualities)
    # print the index with the lowest quality
    print(np.argmin(qualities))
    print("Lowest quality: ", np.min(qualities))


if __name__ == "__main__":
    main()