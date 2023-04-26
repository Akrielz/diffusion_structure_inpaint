from biotite.structure import AtomArray
from biotite.structure import check_bond_continuity as cbc
from biotite.structure.io.pdb import PDBFile
import pdbfixer
from openmm.app import PDBFile
import modeller as m
import modeller.automodel as am
from set_up_modeller_prereqs import extract_chain_to_fasta, run_mmseqs2, process_m8_file, trim_alignment_to_target
from run_modeller import reorder_alignment_file, build_homology_model

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
    
    PDBFile.writeFile(fixer.topology, fixer.positions, open(f'{pdb_file}', 'w'))
    
    return pdb_fixed


def get_alignment(pdb_file: str, target: str, chain_id: str):
    pdbfasta_folder = './mmseqs2/pdbfasta'
    target_folder = './pdb/modeller'
    extract_chain_to_fasta(pdb_file, chain_id, f'{target}.fasta', pdb_file)
    
    target_m8 =  os.path.join(target_folder, f"{target_structure}_{target_chain}.pdbfasta.m8")
    run_mmseqs2(
        target_fasta,
        os.path.join(pdbfasta_folder, 'pdb_seqres_filtered.db'),
        target_m8, 
        iterations=3, 
        skip_if_exists=True
    )
    process_m8_file(
        target_m8, 
        target, 
        chain_id, 
        top_hits=10, 
        identity_threshold=0.9, 
        skip_if_exists=True
    )
    
    target_ali_fasta = os.path.join(target_folder, f"{target}_{chain_id}", f"{target}_{chain_id}_aligned.fasta")
    trim_alignment_to_target(target_ali_fasta, target, chain_id, skip_if_exists=True)
    
    return target_ali_fasta


def homology_model(pdb_file: str, n: int = 5):
    target = pdb_file.split('_')[0]
    chain_id = pdb_file.split('_')[1][0]
    align_file = get_alignment(pdb_file, target, chain_id)
    
    output_folder = 'out'
    build_homology_model(align_file, target, chain, output_folder)

    return f'{output_folder}/{pdb_file}'


def inpainting(pdb_file: str, sequence: str):
    return


def pipeline(pdb_file: str, sequence: str):
    struct = PDBFile.read(pdb_file).get_structure(model=1)
    if not check_discontinuity(struct):
        return struct
    
    pdb_fixed = fix_pdb(pdb_file)
    struct_fixed = PDBFile.read(pdb_fixed).get_structure(model=1)
    if not check_discontinuity(pdb_fixed):
        return struct_fixed

    pdb_templated = homology_model(pdb_fixed, sequence)
    struct_templated = PDBFile.read(pdb_templated).get_structure(model=1)
    if not check_discontinuity(struct_templated):
        return struct_templated

    pdb_inpainted = inpainting(pdb_templated, sequence)
    struct_inpainted = PDBFile.read(pdb_inpainted).get_structure(model=1)
    if not check_discontinuity(struct_inpainted):
        return struct_inpainted
    
    return -1
