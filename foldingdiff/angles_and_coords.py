"""
Code to convert from angles between residues to XYZ coordinates. 
"""
import functools
import gzip
import math
import os
import logging
import glob
from collections import namedtuple, defaultdict
from itertools import groupby
from typing import *
import warnings

import biotite
import numpy
import numpy as np
import pandas as pd

import biotite.structure as struc
import torch
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile
from biotite.sequence import ProteinSequence

from binaries.structure_utils import gradient_descent_on_physical_constraints, write_structure_to_pdb, filter_backbone
from foldingdiff import nerf

EXHAUSTIVE_ANGLES = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]
EXHAUSTIVE_DISTS = ["0C:1N", "N:CA", "CA:C"]

MINIMAL_ANGLES = ["phi", "psi", "omega"]
MINIMAL_DISTS = []


def canonical_distances_and_dihedrals(
    fname: str,
    distances: List[str] = MINIMAL_DISTS,
    angles: List[str] = MINIMAL_ANGLES,
) -> Optional[pd.DataFrame]:
    """Parse the pdb file for the given values"""
    assert os.path.isfile(fname)
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")
    warnings.filterwarnings("ignore", ".*invalid value encountered in true_div.*")
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        source = PDBFile.read(f)
    if source.get_model_count() > 1:
        return None
    # Pull out the atomarray from atomarraystack
    source_struct = source.get_structure()[0]

    # First get the dihedrals
    try:
        phi, psi, omega = struc.dihedral_backbone(source_struct)
        calc_angles = {"phi": phi, "psi": psi, "omega": omega}
    except struc.BadStructureError:
        logging.debug(f"{fname} contains a malformed structure - skipping")
        return None

    # Get any additional angles
    non_dihedral_angles = [a for a in angles if a not in calc_angles]
    # Gets the N - CA - C for each residue
    # https://www.biotite-python.org/apidoc/biotite.structure.filter_backbone.html
    backbone_atoms = source_struct[filter_backbone(source_struct)]
    for a in non_dihedral_angles:
        if a == "tau" or a == "N:CA:C":
            # tau = N - CA - C internal angles
            idx = np.array(
                [list(range(i, i + 3)) for i in range(3, len(backbone_atoms), 3)]
                + [(0, 0, 0)]
            )
        elif a == "CA:C:1N":  # Same as C-N angle in nerf
            # This measures an angle between two residues. Due to the way we build
            # proteins out later, we do not need to meas
            idx = np.array(
                [(i + 1, i + 2, i + 3) for i in range(0, len(backbone_atoms) - 3, 3)]
                + [(0, 0, 0)]
            )
        elif a == "C:1N:1CA":
            idx = np.array(
                [(i + 2, i + 3, i + 4) for i in range(0, len(backbone_atoms) - 3, 3)]
                + [(0, 0, 0)]
            )
        else:
            raise ValueError(f"Unrecognized angle: {a}")
        calc_angles[a] = struc.index_angle(backbone_atoms, indices=idx)

    # At this point we've only looked at dihedral and angles; check value range
    for k, v in calc_angles.items():
        if not (np.nanmin(v) >= -np.pi and np.nanmax(v) <= np.pi):
            logging.warning(f"Illegal values for {k} in {fname} -- skipping")
            return None

    # Get any additional distances
    for d in distances:
        if (d == "0C:1N") or (d == "C:1N"):
            # Since this is measuring the distance between pairs of residues, there
            # is one fewer such measurement than the total number of residues like
            # for dihedrals. Therefore, we pad this with a null 0 value at the end.
            idx = np.array(
                [(i + 2, i + 3) for i in range(0, len(backbone_atoms) - 3, 3)]
                + [(0, 0)]
            )
        elif d == "N:CA":
            # We start resconstructing with a fixed initial residue so we do not need
            # to predict or record the initial distance. Additionally we pad with a
            # null value at the end
            idx = np.array(
                [(i, i + 1) for i in range(3, len(backbone_atoms), 3)] + [(0, 0)]
            )
            assert len(idx) == len(calc_angles["phi"])
        elif d == "CA:C":
            # We start reconstructing with a fixed initial residue so we do not need
            # to predict or record the initial distance. Additionally, we pad with a
            # null value at the end.
            idx = np.array(
                [(i + 1, i + 2) for i in range(3, len(backbone_atoms), 3)] + [(0, 0)]
            )
            assert len(idx) == len(calc_angles["phi"])
        else:
            raise ValueError(f"Unrecognized distance: {d}")
        calc_angles[d] = struc.index_distance(backbone_atoms, indices=idx)

    try:
        return pd.DataFrame({k: calc_angles[k].squeeze() for k in distances + angles})
    except ValueError:
        logging.debug(f"{fname} contains a malformed structure - skipping")
        return None


def compute_coords_from_all(phi, psi, omega, tau, CAC1N, C1NCA):
    bond_lengths = {'N_CA': 1.458, 'CA_C': 1.525, 'C_N+1': 1.330}
    N_CA_unit = np.array([math.cos(phi), math.sin(phi), 0])
    CA_C_unit = np.array([math.cos(psi), math.sin(psi), 0])
    C_N1_unit = np.array([math.cos(omega), math.sin(omega), 0])
    normal_vector = np.cross(CA_C_unit, N_CA_unit)
    R1 = np.array([[math.cos(tau), -math.sin(tau), 0], [math.sin(tau), math.cos(tau), 0], [0, 0, 1]])
    R2 = np.array([[1, 0, 0], [0, math.cos(CAC1N), -math.sin(CAC1N)], [0, math.sin(CAC1N), math.cos(CAC1N)]])
    R3 = np.array([[math.cos(C1NCA), -math.sin(C1NCA), 0], [math.sin(C1NCA), math.cos(C1NCA), 0], [0, 0, 1]])
    C_N1 = -1 * bond_lengths['C_N+1'] * C_N1_unit
    N = np.array([0, 0, 0])
    CA = N + bond_lengths['N_CA'] * N_CA_unit
    C = CA + bond_lengths['CA_C'] * CA_C_unit
    C_N1_rotated = np.dot(R1, C_N1)
    N_rotated = np.dot(R2, N)
    CA_rotated = np.dot(R2, CA)
    C_rotated = np.dot(R3, C)

    coords = {
        'N': N_rotated,
        'CA': CA_rotated,
        'C': C_rotated,
    }
    return coords


def compute_coords(
        phi: float,
        psi: float,
        omega: float
):
    """
    Compute the coordinates of the CA, C, and N atoms of a residue
    based on its phi, psi, and omega angles.

    Parameters
    ----------
    phi : float
        The phi angle (in degrees).
    psi : float
        The psi angle (in degrees).
    omega : float
        The omega angle (in degrees).

    Returns
    -------
    coords : dict of Vector objects
        A dictionary with keys 'CA', 'C', and 'N', each pointing to
        a Vector object containing the coordinates of the respective atom.
    """
    phi, psi, omega = np.radians([phi, psi, omega])
    N_coords = np.array([-np.sin(phi), np.cos(phi), 0])
    CA_coords = np.array([np.cos(psi), np.sin(psi), 0])
    C_coords = np.array([-np.cos(omega), -np.sin(omega), 0])
    coords = {'N': N_coords, 'CA': CA_coords, 'C': C_coords}
    return coords


def combine_original_with_predicted_structure(
        original_atom_array: AtomArray,
        replaced_info_mask: np.ndarray,
        nerf_coords: np.ndarray
):
    """
    Convert the phi, psi, and omega angles of a protein into atomic coordinates.

    Parameters
    ----------
    original_atom_array : AtomArray
        The original AtomArray from the PDB file, without the angle changes.

    replaced_info_mask : torch.Tensor
        A binary tensor of shape (n_residues,), where a 1 indicates that the corresponding
        residue has updated angles.

    nerf_coords : np.ndarray
        The coordinates of the NERF model, of shape (n_residues, 3, 3). The atoms are ordered
        as N, CA, C.

    Returns
    -------
    new_atom_array : AtomArray
        The new AtomArray with updated coordinates based on the new angles.
    """

    from biotite.structure import array as struc_array

    new_residues = []
    for i, residue in enumerate(biotite.structure.residue_iter(original_atom_array)):

        if replaced_info_mask[i] == 0:
            new_residues.append(residue)
            continue

        # Take the nerf coordinates
        coords = {
            'N': nerf_coords[3*i],
            'CA': nerf_coords[3*i+1],
            'C': nerf_coords[3*i+2],
        }

        # Create new atom with the updated coordinates
        ca_atom = biotite.structure.Atom(
            coord=coords['CA'],
            chain_id=residue[0].chain_id,
            res_id=residue[0].res_id,
            ins_code=residue[0].ins_code,
            res_name=residue[0].res_name,
            hetero=residue[0].hetero,
            atom_name='CA',
            element='C',
        )

        c_atom = biotite.structure.Atom(
            coord=coords['C'],
            chain_id=residue[0].chain_id,
            res_id=residue[0].res_id,
            ins_code=residue[0].ins_code,
            res_name=residue[0].res_name,
            hetero=residue[0].hetero,
            atom_name='C',
            element='C',
        )

        n_atom = biotite.structure.Atom(
            coord=coords['N'],
            chain_id=residue[0].chain_id,
            res_id=residue[0].res_id,
            ins_code=residue[0].ins_code,
            res_name=residue[0].res_name,
            hetero=residue[0].hetero,
            atom_name='N',
            element='N',
        )

        residue = [n_atom, ca_atom, c_atom]
        new_residue = struc_array(residue)
        new_residues.append(new_residue)

    atom_arrays = []
    for i, residue in enumerate(new_residues):
        for atom in residue:
            atom_arrays.append(atom)

    new_atom_array = struc_array(atom_arrays)

    return new_atom_array


def create_corrected_structure(
        output_file: str,
        angles: pd.DataFrame,
        initial_atom_array: AtomArray,
        replaced_info_mask: torch.Tensor,
) -> str:
    # Convert the replaced info mask to a numpy array
    replaced_info_mask = replaced_info_mask.cpu().numpy().astype(bool)

    # Extract the already_given_coords for the NERF model
    already_given_coords = extract_backbone_from_struct(initial_atom_array, replaced_info_mask)

    # Create nerf chain
    nerf_coords = create_nerf_chain(
        dists_and_angles=angles,
        center_coords=False,
        already_given_coords=already_given_coords,
        to_generate_mask=replaced_info_mask,
    )

    # Combine the old and new coordinates
    new_atom_array = combine_original_with_predicted_structure(
        original_atom_array=initial_atom_array,
        replaced_info_mask=replaced_info_mask,
        nerf_coords=nerf_coords,
    )

    # Write the new structure to a PDB file
    write_structure_to_pdb(new_atom_array, output_file)
    return output_file


def extract_backbone_from_struct(initial_atom_array, replaced_info_mask):
    already_given_coords = []
    for i, residue in enumerate(biotite.structure.residue_iter(initial_atom_array)):
        if replaced_info_mask[i] == 0:
            # extract the N, CA and C coordinates
            n_coords = residue[residue.atom_name == 'N'][0].coord
            ca_coords = residue[residue.atom_name == 'CA'][0].coord
            c_coords = residue[residue.atom_name == 'C'][0].coord
            already_given_coords.extend(np.array([n_coords, ca_coords, c_coords]))
        else:
            for i in range(3):
                already_given_coords.append(np.zeros(shape=[3]))
    already_given_coords = np.array(already_given_coords)

    return already_given_coords


def create_nerf_chain(
        dists_and_angles: pd.DataFrame,
        angles_to_set: Optional[List[str]] = None,
        dists_to_set: Optional[List[str]] = None,
        center_coords: bool = True,
        already_given_coords: Optional[np.ndarray] = None,
        to_generate_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    angles_to_set, dists_to_set = compute_default_angles_dist_set(angles_to_set, dists_and_angles, dists_to_set)

    # Check that we are at least setting the dihedrals
    required_dihedrals = ["phi", "psi", "omega"]
    assert all([a in angles_to_set for a in required_dihedrals])

    nerf_build_kwargs = compute_nerf_kwargs(
        angles_to_set, dists_and_angles, dists_to_set, required_dihedrals,
        already_given_coords, to_generate_mask
    )

    nerf_builder = nerf.NERFBuilder(**nerf_build_kwargs)
    coords = (
        nerf_builder.centered_cartesian_coords
        if center_coords
        else nerf_builder.cartesian_coords
    )
    return coords


def create_new_chain_nerf_to_file(
        out_fname: str,
        dists_and_angles: pd.DataFrame,
        angles_to_set: Optional[List[str]] = None,
        dists_to_set: Optional[List[str]] = None,
        center_coords: bool = True,
        already_given_coords: Optional[np.ndarray] = None,
        to_generate_mask: Optional[np.ndarray] = None,
) -> str:
    """
    Create a new chain using NERF to convert to cartesian coordinates. Returns
    the path to the newly create file if successful, empty string if fails.
    """
    coords = create_nerf_chain(
        dists_and_angles,
        angles_to_set,
        dists_to_set,
        center_coords,
        already_given_coords,
        to_generate_mask,
    )

    if np.any(np.isnan(coords)):
        logging.warning(f"Found NaN values, not writing pdb file {out_fname}")
        return ""

    assert coords.shape == (
        int(dists_and_angles.shape[0] * 3),
        3,
    ), f"Unexpected shape: {coords.shape} for input of {len(dists_and_angles)}"
    return write_coords_to_pdb(coords, out_fname)


def compute_default_angles_dist_set(angles_to_set, dists_and_angles, dists_to_set):
    if angles_to_set is None and dists_to_set is None:
        angles_to_set, dists_to_set = [], []
        for c in dists_and_angles.columns:
            # Distances are always specified using one : separating two atoms
            # Angles are defined either as : separating 3+ atoms, or as names
            if c.count(":") == 1:
                dists_to_set.append(c)
            else:
                angles_to_set.append(c)
        logging.debug(f"Auto-determined setting {dists_to_set, angles_to_set}")
    else:
        assert angles_to_set is not None
        assert dists_to_set is not None

    return angles_to_set, dists_to_set


def compute_nerf_kwargs(angles_to_set, dists_and_angles, dists_to_set, required_dihedrals, already_given_coords=None, to_generate_mask=None):
    nerf_build_kwargs = dict(
        phi_dihedrals=dists_and_angles["phi"],
        psi_dihedrals=dists_and_angles["psi"],
        omega_dihedrals=dists_and_angles["omega"],
    )
    for a in angles_to_set:
        if a in required_dihedrals:
            continue
        assert a in dists_and_angles
        if a == "tau" or a == "N:CA:C":
            nerf_build_kwargs["bond_angle_ca_c"] = dists_and_angles[a]
        elif a == "CA:C:1N":
            nerf_build_kwargs["bond_angle_c_n"] = dists_and_angles[a]
        elif a == "C:1N:1CA":
            nerf_build_kwargs["bond_angle_n_ca"] = dists_and_angles[a]
        else:
            raise ValueError(f"Unrecognized angle: {a}")
    for d in dists_to_set:
        assert d in dists_and_angles.columns
        if d == "0C:1N":
            nerf_build_kwargs["bond_len_c_n"] = dists_and_angles[d]
        elif d == "N:CA":
            nerf_build_kwargs["bond_len_n_ca"] = dists_and_angles[d]
        elif d == "CA:C":
            nerf_build_kwargs["bond_len_ca_c"] = dists_and_angles[d]
        else:
            raise ValueError(f"Unrecognized distance: {d}")

    if already_given_coords is not None:
        nerf_build_kwargs["already_given_coords"] = already_given_coords
    if to_generate_mask is not None:
        nerf_build_kwargs["to_generate_mask"] = to_generate_mask

    return nerf_build_kwargs


def write_coords_to_pdb(coords: np.ndarray, out_fname: str) -> str:
    """
    Write the coordinates to the given pdb fname
    """
    # Create a new PDB file using biotite
    # https://www.biotite-python.org/tutorial/target/index.html#creating-structures
    assert len(coords) % 3 == 0, f"Expected 3N coords, got {len(coords)}"
    atoms = []
    for i, (n_coord, ca_coord, c_coord) in enumerate(
        (coords[j : j + 3] for j in range(0, len(coords), 3))
    ):
        atom1 = struc.Atom(
            n_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 1,
            res_name="GLY",
            atom_name="N",
            element="N",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom2 = struc.Atom(
            ca_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 2,
            res_name="GLY",
            atom_name="CA",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom3 = struc.Atom(
            c_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 3,
            res_name="GLY",
            atom_name="C",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atoms.extend([atom1, atom2, atom3])
    full_structure = struc.array(atoms)

    # Add bonds
    full_structure.bonds = struc.BondList(full_structure.array_length())
    indices = list(range(full_structure.array_length()))
    for a, b in zip(indices[:-1], indices[1:]):
        full_structure.bonds.add_bond(a, b, bond_type=struc.BondType.SINGLE)

    # Annotate secondary structure using CA coordinates
    # https://www.biotite-python.org/apidoc/biotite.structure.annotate_sse.html
    # https://academic.oup.com/bioinformatics/article/13/3/291/423201
    # a = alpha helix, b = beta sheet, c = coil
    # ss = struc.annotate_sse(full_structure, "A")
    # full_structure.set_annotation("secondary_structure_psea", ss)

    sink = PDBFile()
    sink.set_structure(full_structure)
    sink.write(out_fname)
    return out_fname


@functools.lru_cache(maxsize=8192)
def get_pdb_length(fname: str) -> int:
    """
    Get the length of the chain described in the PDB file
    """
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")
    structure = PDBFile.read(fname)
    if structure.get_model_count() > 1:
        return -1
    chain = structure.get_structure()[0]
    backbone = chain[filter_backbone(chain)]
    l = int(len(backbone) / 3)
    return l


def extract_backbone_coords(
    fname: str, atoms: Collection[Literal["N", "CA", "C"]] = ["CA"]
) -> Optional[np.ndarray]:
    """Extract the coordinates of the alpha carbons"""
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        structure = PDBFile.read(f)
    if structure.get_model_count() > 1:
        return None
    chain = structure.get_structure()[0]
    backbone = chain[filter_backbone(chain)]
    ca = [c for c in backbone if c.atom_name in atoms]
    coords = np.vstack([c.coord for c in ca])
    return coords


SideChainAtomRelative = namedtuple(
    "SideChainAtom", ["name", "element", "bond_dist", "bond_angle", "dihedral_angle"]
)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Gets the angle between u and v"""
    # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    unit_vector = lambda vector: vector / np.linalg.norm(vector)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def collect_aa_sidechain_angles(
    ref_fname: str,
) -> Dict[str, List[SideChainAtomRelative]]:
    """
    Collect the sidechain distances/angles/dihedrals for all amino acids such that
    we can reconstruct an approximate version of them from the backbone coordinates
    and these relative distances/angles/dihedrals

    Returns a dictionary that maps each amino acid residue to a list of SideChainAtom
    objects
    """
    opener = gzip.open if ref_fname.endswith(".gz") else open
    with opener(ref_fname, "rt") as f:
        structure = PDBFile.read(f)
    if structure.get_model_count() > 1:
        raise ValueError
    chain = structure.get_structure()[0]
    retval = defaultdict(list)
    for _, res_atoms in groupby(chain, key=lambda a: a.res_id):
        res_atoms = struc.array(list(res_atoms))
        # Residue name, 3 letter -> 1 letter
        try:
            residue = ProteinSequence.convert_letter_3to1(res_atoms[0].res_name)
        except KeyError:
            logging.warning(
                f"{ref_fname}: Skipping unknown residue {res_atoms[0].res_name}"
            )
            continue
        if residue in retval:
            continue
        backbone_mask = filter_backbone(res_atoms)
        a, b, c = res_atoms[backbone_mask].coord  # Backbone
        for sidechain_atom in res_atoms[~backbone_mask]:
            d = sidechain_atom.coord
            retval[residue].append(
                SideChainAtomRelative(
                    name=sidechain_atom.atom_name,
                    element=sidechain_atom.element,
                    bond_dist=np.linalg.norm(d - c, 2),
                    bond_angle=angle_between(d - c, b - c),
                    dihedral_angle=struc.dihedral(a, b, c, d),
                )
            )
    logging.info(
        "Collected {} amino acid sidechain angles from {}".format(
            len(retval), os.path.abspath(ref_fname)
        )
    )
    return retval


@functools.lru_cache(maxsize=32)
def build_aa_sidechain_dict(
    reference_pdbs: Optional[Collection[str]] = None,
) -> Dict[str, List[SideChainAtomRelative]]:
    """
    Build a dictionary that maps each amino acid residue to a list of SideChainAtom
    that specify how to build out that sidechain's atoms from the backbone
    """
    if not reference_pdbs:
        glob.glob(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/*.pdb")
        )

    retval = {}
    for pdb in reference_pdbs:
        try:
            sidechain_angles = collect_aa_sidechain_angles(pdb)
            retval.update(sidechain_angles)  # Overwrites any existing key/value pairs
        except ValueError:
            continue
    logging.info(f"Built sidechain dictionary with {len(retval)} amino acids")
    return retval


def add_sidechains_to_backbone(
    backbone_pdb_fname: str,
    aa_seq: str,
    out_fname: str,
    reference_pdbs: Optional[Collection[str]] = None,
) -> str:
    """
    Add the sidechains specified by the amino acid sequence to the backbone
    """
    opener = gzip.open if backbone_pdb_fname.endswith(".gz") else open
    with opener(backbone_pdb_fname, "rt") as f:
        structure = PDBFile.read(f)
    if structure.get_model_count() > 1:
        raise ValueError
    chain = structure.get_structure()[0]

    aa_library = build_aa_sidechain_dict(reference_pdbs)

    atom_idx = 1  # 1-indexed
    full_atoms = []
    for res_aa, (_, backbone_atoms) in zip(
        aa_seq, groupby(chain, key=lambda a: a.res_id)
    ):
        backbone_atoms = struc.array(list(backbone_atoms))
        assert len(backbone_atoms) == 3
        for b in backbone_atoms:
            b.atom_id = atom_idx
            atom_idx += 1
            b.res_name = ProteinSequence.convert_letter_1to3(res_aa)
            full_atoms.append(b)
        # Place each atom in the sidechain
        a, b, c = backbone_atoms.coord
        for rel_atom in aa_library[res_aa]:
            d = nerf.place_dihedral(
                a,
                b,
                c,
                rel_atom.bond_angle,
                rel_atom.bond_dist,
                rel_atom.dihedral_angle,
            )
            atom = struc.Atom(
                d,
                chain_id=backbone_atoms[0].chain_id,
                res_id=backbone_atoms[0].res_id,
                atom_id=atom_idx,
                res_name=ProteinSequence.convert_letter_1to3(res_aa),
                atom_name=rel_atom.name,
                element=rel_atom.element,
                hetero=backbone_atoms[0].hetero,
            )
            atom_idx += 1
            full_atoms.append(atom)
    sink = PDBFile()
    sink.set_structure(struc.array(full_atoms))
    sink.write(out_fname)
    return out_fname

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_reverse_dihedral()
    # backbone = collect_aa_sidechain_angles(
    #     os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/1CRN.pdb")
    # )
    # print(build_aa_sidechain_dict())
