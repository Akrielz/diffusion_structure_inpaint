from biotite.structure import AtomArray
from biotite.structure import check_bond_continuity as cbc
from biotite.structure.io.pdb import PDBFile
import pdbfixer
from openmm.app import PDBFile
import modeller as m
import modeller.automodel as am


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


def get_template(sequence: str):
    pass


def get_alignment(pdb_file: str, template_file: str):
    pass


def homology_model(pdb_file: str, sequence: str, n: int = 5)
    env = m.Environ()
    template_pdb = get_template(sequence)
    align_file = get_alignment(pdb_file, template_pdb)
    
    model = am.AutoModel(
                env, 
                alnfile=align_file, 
                knowns=template_pdb, 
                sequence=pdb_file[:-4],
                assess_methods=(am.assess.DOPE),
                inifile=pdb_file
            )
    model.starting_model = 1
    model.ending_model = n
    
    model.make()
    results = [res for res in model.outputs if res['failure'] is None]
    results.sort(key=lambda a: a['DOPE score'])

    return results[0]['name']


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
