import gzip
from collections import defaultdict
from typing import Optional, List, Dict, Any

import biotite
import numpy as np
from Bio import Align
from Bio.pairwise2 import Alignment
from biotite.structure import AtomArray, filter_backbone
from biotite.structure.io.pdb import PDBFile


d3to1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}


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

    stop_pointer = 0
    for i, line in enumerate(source.lines):
        if line.startswith("ATOM"):
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


def determine_missing_residues(pdb_file: str) -> Dict[str, List[int]]:

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
        structure = reindex_aligments(missing_residues_header, structure)

        # We need to drop all the aligments which include already existent residues
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


def reindex_aligments(missing_residues_header, structure):
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


def main():
    file_name = "pdb_to_correct/5f3b.pdb"
    # file_name = "pdb_to_correct/2ZJR_W_broken.pdb"
    # file_name = "pdb_to_correct/6fp7.pdb"

    structure = read_pdb_file(file_name)

    missing_residues_heuristic = determine_missing_residues(file_name)

    print(order_keys_in_dict(missing_residues_heuristic))
    print(get_num_residues_backbone_by_chains(structure))


if __name__ == "__main__":
    main()