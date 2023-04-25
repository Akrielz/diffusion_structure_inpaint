import os
import sys
from modeller import *
from modeller.automodel import *


from Bio import SeqIO

def reorder_alignment_file(input_alignment_file: str, target_name: str, chain: str, output_alignment_file: str):
    """
    Reorder an alignment file, placing the target sequence at the end of the file.

    Args:
        input_alignment_file (str): The path to the input alignment file.
        target_name (str): The target protein name (e.g., "5f3b").
        chain (str): The target protein chain (e.g., "C").
        output_alignment_file (str): The path to the output reordered alignment file.

    Returns:
        bool: True if the operation was successful or False otherwise, and a message describing the result.
    """
    target_code = f"{target_name}_{chain}"
    
    # Read and parse the input alignment file using Biopython
    records = list(SeqIO.parse(input_alignment_file, "fasta"))
    
    # Separate the target sequence from the other sequences
    target_record = None
    other_records = []
    for record in records:
        if record.id == target_code:
            target_record = record
        else:
            other_records.append(record)
    
    # Check if the target sequence was found
    if not target_record:
        print(f"Target sequence '{target_code}' not found in the input alignment file.")
        return False
    
    # Write the reordered alignment file
    with open(output_alignment_file, "w") as output_file:
        SeqIO.write(other_records + [target_record], output_file, "fasta")
    
    print("Reordered alignment file created successfully.")
    return True

# Just testing for now
def build_homology_model(alignment_file, target_name, chain, output_folder):
    # Get the template PDB codes

    aln = Alignment(env)
    aln.read(file=alignment_file, alignment_format='PIR')

    target_code = f"{target_name}_{chain}"
    template_codes = [prot.code for prot in aln if prot.code != target_code]
    print(target_code, template_codes)

    # Create a new automodel object
    class MyModel(AutoModel):
        def select_atoms(self):
            return selection(self.residue_range(f'1:{chain}', f'last:{chain}'))

    mdl = AutoModel(env, alnfile=alignment_file, knowns=template_codes, sequence=target_code)
    mdl.starting_model = 1
    mdl.ending_model = 1

    # Set the output directory
    mdl.path = output_folder

    # Build the model
    mdl.make()

env = Environ()
# directories for input atom files
env.io.atom_files_directory = ['.', '5f3b_C', 'pdb/modeller/5f3b_C']

alignment_file = 'pdb/modeller/5f3b_C/5f3b_C_aligned_trimmed.fasta'
reordered_alignment_file = 'pdb/modeller/5f3b_C/5f3b_C_aligned_trimmed_reordered.fasta'
target_name = '5f3b'
chain = 'C'
output_folder = '5f3b_C/'

# reorder_alignment_file(alignment_file, target_name, chain, reordered_alignment_file)
reordered_alignment_file = 'pdb/modeller/5f3b_C/sample_new.pir'
build_homology_model(reordered_alignment_file, target_name, chain, output_folder)

