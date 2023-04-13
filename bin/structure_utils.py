import gzip
from collections import defaultdict
from typing import Optional, List, Dict

import biotite
import numpy as np
from Bio import pairwise2
from Bio.pairwise2 import format_alignment, Alignment
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile


d3to1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
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
    return len(set(structure.res_id))


def get_num_residues_by_chains(structure: AtomArray) -> Dict[str, int]:
    chains = get_chains(structure)
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


def sequence_alignment(header_seq: str, structure_seq: str) -> Alignment:
    alignments = pairwise2.align.globalxx(header_seq, structure_seq)
    alignment = alignments[0]

    return alignment


def missing_residues_by_sequence_alignment(header_seq: str, structure_seq: str) -> List[int]:

    alignment = sequence_alignment(header_seq, structure_seq)

    # Determine the missing residues
    missing_residues = []
    for i, (header_res, structure_res) in enumerate(zip(alignment[0], alignment[1])):
        if header_res != structure_res:
            missing_residues.append(i+1)

    return missing_residues


def missing_residues_by_sequence_alignment_by_chains(header_seq: Dict[str, str], structure_seq: Dict[str, str]) -> Dict[str, List[int]]:
    missing_residues_by_chains = {}
    for chain in header_seq.keys():
        missing_residues_by_chains[chain] = missing_residues_by_sequence_alignment(
            header_seq[chain], structure_seq[chain]
        )

    return missing_residues_by_chains


def missing_residues_by_structure_look_residues_id(structure: AtomArray) -> Dict[str, List[int]]:
    chains = get_chains(structure)
    structure = filter_only_ca(structure)
    missing_residues_by_chains = {}
    for chain in chains:
        missing_residues_by_chains[chain] = []
        for i in range(1, get_num_residues(structure[structure.chain_id == chain]) + 1):
            if len(structure[(structure.chain_id == chain) & (structure.res_id == i)]) == 0:
                missing_residues_by_chains[chain].append(i)

    return missing_residues_by_chains


def missing_residues_by_structure_continuity(structure: AtomArray) -> Dict[str, List[int]]:
    chains = get_chains(structure)
    missing_residues_by_chains = {}
    for chain in chains:
        chain_structure = structure[(structure.chain_id == chain)]
        missing_residues: np.ndarray = biotite.structure.check_bond_continuity(chain_structure)
        missing_residues = missing_residues // 3
        missing_residues_by_chains[chain]: List[int] = missing_residues.tolist()

    return missing_residues_by_chains


def main():
    file_name = "pdb_to_correct/5f3b.pdb"
    header = read_pdb_header(file_name)
    header_seq = get_sequence_from_header(header)

    structure = read_pdb_file(file_name)
    structure_seq = get_sequence_from_structure(structure)

    missing_residues_header = missing_residues_by_sequence_alignment_by_chains(header_seq, structure_seq)
    missing_residues_struct_res = missing_residues_by_structure_look_residues_id(structure)
    missing_residues_struct_cont = missing_residues_by_structure_continuity(structure)

    print(missing_residues_header)
    print(missing_residues_struct_res)
    print(missing_residues_struct_cont)


if __name__ == "__main__":
    main()