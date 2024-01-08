import os
import csv
import gzip
import zipfile
import subprocess
import pandas as pd
from tqdm import tqdm
import urllib.request



def download_gbif_taxonomy(output_folder=None):
    """
    Download the GBIF backbone taxonomy dataset and save the 'Taxon.tsv' file in a 'GBIF_output' folder.

    Args:
    output_folder (str, optional): The folder where the dataset will be downloaded. 
                                   Defaults to the current working directory.
    """
    
    if output_folder is None:
        output_folder = os.getcwd()

    # Create GBIF_output folder inside the specified output folder
    gbif_output_folder = os.path.join(output_folder, 'GBIF_output')
    os.makedirs(gbif_output_folder, exist_ok=True)

    url = "https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip"
    filename = os.path.join(output_folder, "backbone.zip")  

    # Download the file with a progress bar
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading GBIF Taxonomic Data") as t:
        urllib.request.urlretrieve(url, filename, reporthook=lambda blocknum, blocksize, totalsize: t.update(blocksize))

    # Check if the file was downloaded successfully
    if os.path.exists(filename):
        print("GBIF backbone taxonomy has been downloaded successfully.")

    # Extract the TSV file to the GBIF_output folder
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extract('Taxon.tsv', gbif_output_folder)

    tsv_path = os.path.join(gbif_output_folder, 'Taxon.tsv')
    if os.path.exists(tsv_path):
        os.remove(filename)  # Remove the zip file after extracting
    else:
        raise RuntimeError("Error saving Taxon.tsv. Consider raising an issue on Github.")


def compare_versions(version1, version2):
    """
    Compare two version strings.

    Args:
    version1 (str): The first version string.
    version2 (str): The second version string.

    Returns:
    int: -1 if version1 < version2, 1 if version1 > version2, 0 if they are equal.
    """
    
    v1_components = list(map(int, version1.split(".")))
    v2_components = list(map(int, version2.split(".")))

    for v1, v2 in zip(v1_components, v2_components):
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1

    return 0



def download_ncbi_taxonomy(taxonkitpath=None, output_folder=None):
    """
    Download the NCBI taxonomy dataset using TaxonKit.

    Args:
    taxonkitpath (str, optional): The path to the TaxonKit binary.
    output_folder (str, optional): The folder where the dataset will be downloaded. Defaults to the current working directory.
    """
    
    if taxonkitpath is not None:
        if "taxonkit" not in taxonkitpath.lower():
            raise ValueError("The path must include both the directory name and filename 'taxonkit'")
        
        try:
            # Check TaxonKit version
            version_output = subprocess.check_output([taxonkitpath, "version"], stderr=subprocess.DEVNULL, text=True)
            version = version_output.strip().replace("taxonkit v", "")
            if compare_versions(version, "0.8.0") == 0:
                raise ValueError("TaxonKit version 0.8.0 or greater is required. Please download a more recent version of TaxonKit and try again.")
        except subprocess.CalledProcessError:
            raise ValueError(f"TaxonKit not detected. Is this the correct path to TaxonKit: {taxonkitpath}?")
    else:
        raise ValueError("Please specify the taxonkit path!")

    if output_folder is None:
        output_folder = os.getcwd()

    # Create a new folder for TaxonKit output
    taxonkit_output_folder = os.path.join(output_folder, "NCBI_output")
    os.makedirs(taxonkit_output_folder, exist_ok=True)

    # Download the NCBI taxondump
    url = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdmp.zip"
    tf = os.path.join(taxonkit_output_folder, "taxdmp.zip")
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading NCBI Taxonomic Data") as pbar:
        def report_hook(blocknum, blocksize, totalsize):
            pbar.update(blocknum * blocksize / totalsize * 100)
        urllib.request.urlretrieve(url, tf, reporthook=report_hook)

    # Decompress the downloaded zip file
    with zipfile.ZipFile(tf, "r") as zip_ref:
        zip_ref.extractall(taxonkit_output_folder)

    gz_path = taxonkit_output_folder + '/All.lineages.tsv.gz'
    file_path = taxonkit_output_folder + '/ncbi_data.tsv'

    # Construct and run the TaxonKit command
    full_command = (
        f"{taxonkitpath} --data-dir={taxonkit_output_folder} list --ids 1 | "
        f"{taxonkitpath} lineage --show-lineage-taxids --show-lineage-ranks --show-rank --show-name --data-dir={taxonkit_output_folder} | "
        f"{taxonkitpath} reformat --taxid-field 1 --data-dir={taxonkit_output_folder} -o {gz_path}"
    )

    try:
        subprocess.run(full_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("NCBI taxonomic data has been downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print("Error:", e)

    # Read the compressed file and write its contents to a TSV file
    with gzip.open(gz_path, 'rb') as gz_file:
        decompressed_content = gz_file.read().decode('utf-8')

    with open(file_path, 'wt') as output_file:
        csv_writer = csv.writer(output_file, delimiter='\t')
        for line in decompressed_content.splitlines():
            row = line.split('\t')
            csv_writer.writerow(row)
