from Bio import AlignIO, BiopythonDeprecationWarning, SeqIO
from Bio.PDB import PDBParser, Polypeptide, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modeller import Environ, Alignment
from modeller.automodel import AutoModel

import os
import subprocess
import tempfile
import urllib
import warnings

warnings.simplefilter('ignore', PDBConstructionWarning)
warnings.simplefilter('ignore', BiopythonDeprecationWarning)


class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id


def make_output_folder(path):
    """
    Creates an output folder including any necessary parent directories if they don't already exist.

    Args:
        path (str): The path to the output folder that should be created.

    Returns:
        bool: True if the folder was created or already exists, False otherwise.
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating output folder: {e}")
        return False


def extract_chain_to_fasta(pdb_file: str, chain_id: str, output_fasta: str, output_pdb: str, skip_if_exists: bool = True):
    if skip_if_exists and os.path.exists(output_fasta):
        print(f"File '{output_fasta}' already exists. Skipping extraction.")
        return

    parser = PDBParser()
    structure = parser.get_structure('PDB', pdb_file)
    model = structure[0]
    chain = model[chain_id]

    sequence = ""
    prev_residue_number = None

    for residue in chain:
        if residue.id[0] == " ":
            residue_number = residue.id[1]
            if prev_residue_number is not None and residue_number - prev_residue_number > 1:
                sequence += "-" * (residue_number - prev_residue_number - 1)
            sequence += Polypeptide.three_to_one(residue.resname)
            prev_residue_number = residue_number

    seq = Seq(sequence)
    seq_record = SeqRecord(seq)
    seq_record.id = f"{pdb_file[-8:-4]}_{chain_id}"
    seq_record.description = ""

    with open(output_fasta, "w") as output_handle:
        SeqIO.write(seq_record, output_handle, "fasta")

    # Save chain as PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, ChainSelect(chain_id))
    return True

def download_pdb_file(pdb_id: str, output_folder: str, skip_if_exists: bool = True):
    """
    Download the specified PDB file and save it in the given output folder.

    Args:
        pdb_id (str): The PDB ID of the structure to download.
        output_folder (str): The folder in which to save the PDB file.
        skip_if_exists (bool, optional): If True, skip the download if the file already exists.
                                         Defaults to True.

    Returns:
        bool: True if the file is downloaded or already exists, False otherwise.
    """
    make_output_folder(output_folder)
    pdb_file = os.path.join(output_folder, f"{pdb_id}.pdb")

    if skip_if_exists and os.path.exists(pdb_file):
        print(f"File '{pdb_file}' already exists. Skipping download.")
        return True

    command = f"pdb_fetch {pdb_id} > {pdb_file}"
    process = subprocess.run(command, shell=True, capture_output=True, text=True)

    if process.returncode != 0 or os.path.getsize(pdb_file) == 0:
        print(f"Error downloading structure {pdb_id}.")
        os.remove(pdb_file)
        return False

    print(f"Structure {pdb_id} downloaded successfully.")
    return True


def download_and_extract_pdb_seqres(output_folder, output_filename="pdb_seqres_filtered.fasta", skip_if_exists=True):
    """
    Downloads, extracts, filters, and saves protein entries from the pdb_seqres.txt.gz file as a fasta file.
    
    Args:
        output_folder (str): The path to the folder where the output file will be saved.
        output_filename (str): The name of the output fasta file. Default is "pdb_seqres_filtered.fasta".
        skip_if_exists (bool): Whether to skip the download and extraction if the output file already exists.

    Returns:
        bool: True if the operation succeeded and the output file is not empty, False otherwise.
    """
    make_output_folder(output_folder)
    output_path = os.path.join(output_folder, output_filename)

    # Check if output file already exists
    if skip_if_exists and os.path.isfile(output_path):
        print("Output file already exists, skipping download and extraction.")
        return True

    url = "https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
    file_path = os.path.join(output_folder, "pdb_seqres.txt.gz")

    # Download pdb_seqres.txt.gz
    try:
        print("Downloading pdb_seqres.txt.gz")
        urllib.request.urlretrieve(url, file_path)
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

    # Extract and parse pdb_seqres.txt.gz
    proteins = []
    try:
        with gzip.open(file_path, 'rt') as f:
            for record in SeqIO.parse(f, "fasta"):
                if "mol:protein" in record.description:
                    proteins.append(record)
    except Exception as e:
        print(f"Error extracting and parsing file: {e}")
        return False

    # Save filtered protein entries as fasta
    try:
        with open(output_path, "w") as f:
            SeqIO.write(proteins, f, "fasta")
    except Exception as e:
        print(f"Error saving filtered protein entries: {e}")
        return False

    # Check if output file is not empty
    if os.path.getsize(output_path) == 0:
        print("Error: output file is empty.")
        return False

    return True


def create_mmseqs2_database(fasta_file: str, output_folder: str, skip_if_exists: bool = True) -> bool:
    """
    Creates an MMseqs2 database from a given FASTA file and stores it in the specified folder.

    Args:
        fasta_file (str): The path to the FASTA file.
        output_folder (str): The folder where the MMseqs2 database will be created.
        skip_if_exists (bool): Whether to skip the database creation if the output files already exist.

    Returns:
        bool: True if the database is created or already exists, False otherwise.
    """
    make_output_folder(output_folder)

    # Check if the output files already exist
    db_name = os.path.splitext(os.path.basename(fasta_file))[0]
    output_files = [f"{db_name}.db.index", f"{db_name}.db"]

    if skip_if_exists and all(os.path.isfile(os.path.join(output_folder, f)) for f in output_files):
        print("Database files already exist, skipping creation.")
        return True

    # Create the MMseqs2 database
    try:
        command = f"mmseqs createdb {fasta_file} {os.path.join(output_folder, db_name + '.db')}"
        subprocess.run(command, shell=True, check=True)
        print("MMseqs2 database created successfully.")

        # Create the database index
        command = f"mmseqs createindex {os.path.join(output_folder, db_name + '.db')} {output_folder}"
        subprocess.run(command, shell=True, check=True)
        print("MMseqs2 database index created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create the MMseqs2 database and/or index. Error: {e}")
        return False


def run_mmseqs2(query_fasta: str, database: str, output_m8: str, skip_if_exists: bool = True, iterations: int = 3):
    """
    Run MMseqs2 with a specified query FASTA file against a database to generate an alignment (A3M) file.

    Args:
        query_fasta (str): Path to the query FASTA file.
        database (str): Path to the MMseqs2 database folder.
        output_m8 (str): Path to the output M8 hits file.
        skip_if_exists (bool, optional): Whether to skip the search if the output file already exists. Defaults to True.
        iterations (int, optional): Number of MMseqs2 iterations. Defaults to 3.

    Returns:
        bool: True if the hits file exists and is not empty, False otherwise.
    """
    if skip_if_exists and os.path.exists(output_m8) and os.path.getsize(output_m8) > 0:
        print("A3M output file already exists, skipping search.")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        mmseqs_easy_search_cmd = (
            f"mmseqs easy-search {query_fasta} {database} {output_m8} {tmpdir} --format-mode 2 --num-iterations {iterations}"
        )

        return_code = subprocess.call(mmseqs_easy_search_cmd, shell=True)

        return return_code == 0 and os.path.exists(output_m8) and os.path.getsize(output_m8) > 0


def process_m8_file(m8_file: str, target_name: str, chain: str, identity_threshold: float = 0.9, top_hits: int = 15, skip_if_exists: bool = True):
    """
    Parse the m8 file, download the top hits with an identity threshold above a given value, extract the relevant chains,
    and perform the alignment using Clustal Omega.

    Args:
        m8_file (str): Path to the input m8 file.
        target_name (str): The name of the target protein structure.
        chain (str): The identifier of the chain being fixed.
        identity_threshold (float, optional): The minimum identity threshold to consider a hit. Defaults to 0.9.
        top_hits (int, optional): The maximum number of hits to process. Defaults to 15.
        skip_if_exists (bool, optional): If True, skip downloading PDB files or extracting chains if the output files already exist. Defaults to True.

    Returns:
        bool: True if the alignment file exists and is not empty, False otherwise.
    """
    # Parse m8 file
    hits = []
    with open(m8_file, 'r') as file:
        for line in file:
            fields = line.strip().split('\t')
            if len(fields) > 2 and float(fields[2]) >= identity_threshold:
                hits.append((fields[1].split('_')[0], fields[1][-1]))
            if len(hits) >= top_hits:
                break
    
    if len(hits) == 0:
        print("No suitable hits found.")
        return False

    # Create folder named after the chain we are fixing
    m8_file_path = os.path.dirname(os.path.abspath(m8_file))
    output_folder = os.path.join(m8_file_path, f"{target_name}_{chain}")

    alignment_file = os.path.join(output_folder, f"{target_name}_{chain}_aligned.fasta")
    if skip_if_exists and os.path.exists(alignment_file):
        print(f"Alignment file '{alignment_file}' already exists. Skipping alignment.")
        return True

    make_output_folder(output_folder)

    # Download PDB files and extract the chains to FASTA format
    fasta_files = []
    for pdb_id, chain_id in hits:
        if download_pdb_file(pdb_id, output_folder, skip_if_exists=skip_if_exists):
            fasta_file = os.path.join(output_folder, f"{pdb_id}_{chain_id}.fasta")
            pdb_file = os.path.join(output_folder, f"{pdb_id}_{chain_id}.pdb")
            if extract_chain_to_fasta(os.path.join(output_folder, f"{pdb_id}.pdb"), chain_id, fasta_file, pdb_file, skip_if_exists=skip_if_exists):
                fasta_files.append(fasta_file)

    # Concatenate all FASTA files into a single file
    concatenated_fasta_file = os.path.join(output_folder, f"{target_name}_{chain}_concatenated.fasta")
    with open(concatenated_fasta_file, "w") as outfile:
        for fasta_file in fasta_files:
            if os.path.isfile(fasta_file):
                with open(fasta_file, "r") as infile:
                    outfile.write(infile.read())

    # Perform Clustal Omega alignment
    clustalo_command = f"clustalo -i {concatenated_fasta_file} -o {alignment_file} --force"
    process = subprocess.run(clustalo_command, shell=True, capture_output=True, text=True)

    if process.returncode != 0:
        print(f"Error performing Clustal Omega alignment.")
        return False

    return True


def trim_alignment_to_target(alignment_file: str, target_name: str, chain: str, skip_if_exists: bool = True):
    """
    Trim the alignment file to only contain the ranges of the target sequence, i.e., remove all the residues or gaps
    that are outside the target sequence.

    Args:
        alignment_file (str): Path to the input alignment file.
        target_name (str): The name of the target protein structure.
        chain (str): The identifier of the chain being fixed.
        skip_if_exists (bool, optional): If True, skip trimming if the output file already exists. Defaults to True.

    Returns:
        bool: True if the trimmed alignment file exists and is not empty, False otherwise.
    """

    ali_file_path = os.path.dirname(os.path.abspath(alignment_file))
    trimmed_ali_file = os.path.join(ali_file_path, f"{target_name}_{chain}_aligned_trimmed.fasta")

    if skip_if_exists and os.path.exists(trimmed_ali_file):
        print(f"File '{trimmed_ali_file}' already exists. Skipping trimming.")
        return True

    alignment = AlignIO.read(alignment_file, "fasta")
    target_seq = None

    for record in alignment:
        if record.id == f"{target_name}_{chain}":
            target_seq = record
            break

    if target_seq is None:
        print(f"Target sequence '{target_name}_{chain}' not found in the alignment.")
        return False

    start = None
    end = None

    for i, char in enumerate(target_seq.seq):
        if char != '-':
            start = i
            break

    for i, char in enumerate(target_seq.seq[::-1]):
        if char != '-':
            end = len(target_seq.seq) - i
            break

    if start is None or end is None:
        print("Error finding start and end positions of the target sequence.")
        return False

    trimmed_alignment = alignment[:, start:end]
    AlignIO.write(trimmed_alignment, trimmed_ali_file, "fasta")
    return trimmed_ali_file


def convert_to_pir(aln_file: str):
    with open(aln_file) as f:
        lines = f.readlines()

    pir_lines = [c.replace('>', '>P1;') for c in lines]
    s_ind = [i for i, c in enumerate(pir_lines) if '>' in c]
    for i in s_ind:
        pir_lines[i - 1] = pir_lines[i - 1].replace('\n', '*\n')
    for i, j in enumerate(s_ind):
        pir_lines.insert(i + j + 1, f"structure:{pir_lines[i + j].split(';')[-1][:-1]}:.:.:.:.::::\n")
    
    pir = f"{aln_file.split('.')[0]}.pir"
    with open(pir, 'w') as f:
        f.writelines(pir_lines)
    return pir


def reorder_alignment_file(input_alignment_file: str, target_name: str, chain: str, output_alignment_file: str):
    """
    Reorder an alignment file, placing the target sequence at the end of the file.

    Args:
        input_alignment_file (str): The path to the input alignment file.
        target_name (str): The target protein name (e.g., "5f3b").
        chain (str): The target protein chain (e.g., "C").
        output_alignment_file (str): The path to the output reordered alignment file.
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


def get_alignment(pdb_file: str, target: str, chain_id: str):
    pdbfasta_folder = './mmseqs2/pdbfasta'
    download_and_extract_pdb_seqres(pdbfasta_folder)
    create_mmseqs2_database(os.path.join(pdbfasta_folder, 'pdb_seqres_filtered.fasta'), pdbfasta_folder)

    target_folder = './pdb/modeller'
    target_fasta = os.path.join(target_folder, f'{target}_{chain_id}.fasta')
    extract_chain_to_fasta(pdb_file, chain_id, target_fasta, pdb_file)
    
    target_m8 =  os.path.join(target_folder, f"{target}_{chain_id}.pdbfasta.m8")
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
        top_hits=5, 
        identity_threshold=0.9, 
        skip_if_exists=True
    )
    
    target_ali_fasta = os.path.join(target_folder, f"{target}_{chain_id}", f"{target}_{chain_id}_aligned.fasta")
    trimmed_ali_fasta = trim_alignment_to_target(target_ali_fasta, target, chain_id, skip_if_exists=True)
    ali_pir = convert_to_pir(trimmed_ali_fasta)
    return ali_pir


def build_homology_model(alignment_file, target_name, chain):
    env = Environ()
    target_code = f"{target_name}_{chain}"
    env.io.atom_files_directory = ['.', f'{target_code}', f'pdb/modeller/{target_code}']
    
    aln = Alignment(env)
    aln.read(file=alignment_file, alignment_format='PIR')
    template_codes = [prot.code for prot in aln if prot.code != target_code]

    # Create a new automodel object
    class MyModel(AutoModel):
        def select_atoms(self):
            return selection(self.residue_range(f'1:A', f'1:A'))

    mdl = AutoModel(
        env, 
        alnfile=alignment_file, 
        knowns=template_codes, 
        sequence=target_code, 
        inifile=target_code
    )
    mdl.starting_model = 1
    mdl.ending_model = 1
    
    # Build the model
    mdl.make()
    return mdl.outputs[0]['name']