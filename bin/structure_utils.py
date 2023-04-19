import gzip
from collections import defaultdict
from typing import Optional, List, Dict, Any

import biotite
import numpy as np
from Bio import Align
from Bio.pairwise2 import Alignment
from biotite.structure import AtomArray, filter_backbone
from biotite.structure.io.pdb import PDBFile

import faspr

d3to1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}


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
    return min(set(structure.res_id))


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


def sequence_alignment(header_seq: str, structure_seq: str) -> Alignment:
    aligner = Align.PairwiseAligner()
    aligner.gap_score = -1
    aligner.mismatch_score = -1
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(header_seq, structure_seq)
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
        chain_structure = structure[structure.chain_id == chain]

        for i in range(get_lowest_residue(chain_structure), get_num_residues(chain_structure) + 1):
            if len(chain_structure[chain_structure.res_id == i]) == 0:
                missing_residues_by_chains[chain].append(i)

    return missing_residues_by_chains


def missing_residues_by_structure_continuity(structure: AtomArray) -> Dict[str, List[int]]:
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

        for i in range(get_lowest_residue(backbone_structure), get_num_residues(backbone_structure) + 1):
            backbone_residue = chain_structure[chain_structure.res_id == i]
            n_atom = backbone_residue[backbone_residue.atom_name == "N"]
            ca_atom = backbone_residue[backbone_residue.atom_name == "CA"]
            c_atom = backbone_residue[backbone_residue.atom_name == "C"]

            if len(backbone_residue) == 0:
                continue

            if len(n_atom) == 0 or len(ca_atom) == 0 or len(c_atom) == 0:
                broken_residues_by_chains[chain].append(i)

    return broken_residues_by_chains


def model_sidechains(input_file: str, output_file: str, residue_indices: List[int] = []):
    if residue_indices:
        "Indices where residue sidechains need modeling"
        structure = read_pdb_file(input_file)
        struct_seq = get_sequence_from_structure(structure).lower()
        seq = ''.join([c.upper() if i in residue_indices else c for i, c in enumerate(struct_seq)])
        seq_file = f'{input_file[:-4]}.txt' 
        with open(seq_file, 'w') as f:
            f.write(seq)
        seq_flag = True
    else:
        "This will instead repack all the sidechains in the pdb file"
        seq_file = ''
        seq_flag = False

    faspr.FASPRcpp(input_file, output_file, seq_file, seq_flag)

def main():
    file_name = "pdb_to_correct/5f3b.pdb"
    # file_name = "pdb_to_correct/6fp7.pdb"
    header = read_pdb_header(file_name)
    header_seq = get_sequence_from_header(header)

    structure = read_pdb_file(file_name)
    structure_seq = get_sequence_from_structure(structure)

    missing_residues_header = missing_residues_by_sequence_alignment_by_chains(header_seq, structure_seq)
    missing_residues_struct_res = missing_residues_by_structure_look_residues_id(structure)
    missing_residues_struct_cont = missing_residues_by_structure_continuity(structure)
    broken_residues_struct = broken_residues_by_structure(structure)

    print(order_keys_in_dict(missing_residues_header))
    print(order_keys_in_dict(missing_residues_struct_res))
    print(order_keys_in_dict(missing_residues_struct_cont))
    print(order_keys_in_dict(broken_residues_struct))
    print(get_num_residues_backbone_by_chains(structure))


if __name__ == "__main__":
    main()
