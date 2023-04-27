from biotite.structure import AtomArray
from biotite.structure import check_bond_continuity as cbc
from biotite.structure.io.pdb import PDBFile
import pdbfixer
from openmm.app import PDBFile as PDBFileMM
from modeller_utils import get_alignment, build_homology_model


def check_discontinuity(struct: AtomArray):
    disc = cbc(struct)
    return len(disc) > 0


def fix_pdb(pdb_file: str):
    fixer = pdbfixer.PDBFixer(pdb_file)
    
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    PDBFileMM.writeFile(fixer.topology, fixer.positions, open(f'{pdb_file}', 'w'), True)
    
    return pdb_file


def homology_model(pdb_file: str):
    target = pdb_file.split('_')[0]
    chain_id = pdb_file.split('_')[1][0]
    
    align_file = get_alignment(pdb_file, target, chain_id)
    model_file = build_homology_model(align_file, target, chain_id)
    
    return model_file


def inpainting(pdb_file: str, sequence: str):
    return


def pipeline(pdb_file: str):
    struct = PDBFile.read(pdb_file).get_structure(model=1)
    if not check_discontinuity(struct):
        return struct
    
    pdb_fixed = fix_pdb(pdb_file)
    struct_fixed = PDBFile.read(pdb_fixed).get_structure(model=1)
    if not check_discontinuity(struct_fixed):
        return struct_fixed

    pdb_templated = homology_model(pdb_fixed)
    struct_templated = PDBFile.read(pdb_templated).get_structure(model=1)
    if not check_discontinuity(struct_templated):
        return struct_templated

    pdb_inpainted = inpainting(pdb_templated)
    struct_inpainted = PDBFile.read(pdb_inpainted).get_structure(model=1)
    if not check_discontinuity(struct_inpainted):
        return struct_inpainted
    
    return

if __name__ == '__main__':
    pipeline('5f3b_C.pdb')