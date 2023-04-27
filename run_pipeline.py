from biotite.structure import AtomArray
from biotite.structure import check_bond_continuity
from biotite.structure.io.pdb import PDBFile
import pdbfixer
from openmm.app import PDBFile as PDBFileMM
from modeller_utils import get_alignment, build_homology_model
from typing import List

def check_discontinuity(struct: AtomArray):
    disc = check_bond_continuity(struct)
    return len(disc) > 0


def fix_pdb(pdb_file: str):
    fixer = pdbfixer.PDBFixer(pdb_file)
    
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    PDBFileMM.writeFile(fixer.topology, fixer.positions, open(f'{pdb_file[:-4]}_fixed.pdb', 'w'), True)
    
    return pdb_file


def homology_model(pdb_file: str):
    target = pdb_file.split('_')[0]
    chain_id = pdb_file.split('_')[1][0]
    
    align_file = get_alignment(pdb_file, target, chain_id)
    model_file = build_homology_model(align_file, target, chain_id)
    
    return model_file


def inpainting(pdb_file: str, sequence: str):
    return


def batch_inpainting(pdb_files: List[str]):
    passed, failed = [], []

    return passed, failed


def pipeline(pdb_file: str):
    struct = PDBFile.read(pdb_file).get_structure(model=1)
    if not check_discontinuity(struct):
        return struct
    
    pdb_fixed = fix_pdb(pdb_file)
    struct_fixed = PDBFile.read(pdb_fixed).get_structure(model=1)
    if not check_discontinuity(struct_fixed):
        return pdb_fixed

    pdb_templated = homology_model(pdb_fixed)
    struct_templated = PDBFile.read(pdb_templated).get_structure(model=1)
    if not check_discontinuity(struct_templated):
        return pdb_templated

    pdb_inpainted = inpainting(pdb_templated)
    struct_inpainted = PDBFile.read(pdb_inpainted).get_structure(model=1)
    if not check_discontinuity(struct_inpainted):
        return pdb_inpainted
    
    return pdb_file


def batch_pipeline(pdb_files: List[str]):
    passed, failed = [], []
    for pdb_file in pdb_files:
        struct = PDBFile.read(pdb_file).get_structure(model=1)
        if not check_discontinuity(struct):
            passed.append(pdb_file)
            continue
        
        pdb_fixed = fix_pdb(pdb_file)
        struct_fixed = PDBFile.read(pdb_fixed).get_structure(model=1)
        if not check_discontinuity(struct_fixed):
            passed.append(pdb_fixed)
            continue

        pdb_templated = homology_model(pdb_fixed)
        struct_templated = PDBFile.read(pdb_templated).get_structure(model=1)
        if not check_discontinuity(struct_templated):
            passed.append(pdb_templated)
            continue

        failed.append(pdb_templated)
    
    inpainting_passed, failed = batch_inpainting(failed)
    passed.extend(inpainting_passed)
    
    return passed, failed


if __name__ == '__main__':
    pipeline('5f3b_C.pdb')