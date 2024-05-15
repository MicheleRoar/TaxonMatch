import os
import re
import time
import zipfile
import threading
import numpy as np
import pandas as pd
import urllib.request
from tqdm import tqdm




def create_gbif_taxonomy(row):
    """
    Create a taxonomy string for GBIF data.

    Args:
    row (pd.Series): A row from the GBIF DataFrame.

    Returns:
    str: A semicolon-separated string of taxonomy levels.
    """
    if row['taxonRank'] in ['species', 'subspecies', "variety"]:
        return f"{row['phylum']};{row['class']};{row['order']};{row['family']};{row['genus']};{row['canonicalName']}".lower()
    else:
        return f"{row['phylum']};{row['class']};{row['order']};{row['family']};{row['genus']}".lower()

def find_all_parents(taxon_id, parents_dict):
    """
    Find all parent taxon IDs for a given taxon ID.

    Args:
    taxon_id (int): The taxon ID to find parents for.
    parents_dict (dict): A dictionary mapping taxon IDs to their parent taxon IDs.

    Returns:
    str: A semicolon-separated string of parent taxon IDs.
    """
    parents = []
    while taxon_id != -1:  # -1 represents the NaN converted value
        parents.append(taxon_id)
        taxon_id = parents_dict.get(taxon_id, -1)  # Get the parent

    # Reverse the list of parents and create a string separated by ";", excluding the first element
    parents_string = ';'.join(map(str, parents[::-1][1:])) if len(parents) > 1 else ''

    return parents_string

def load_gbif_samples(gbif_path_file):
    """
    Load GBIF samples from a file and process them.

    Args:
    gbif_path_file (str): Path to the GBIF file.

    Returns:
    tuple: A tuple containing the processed subset and the full DataFrame of GBIF data.
    """


    def animate_dots():
        """
        Animate a sequence of dots on the console to indicate processing activity.
            
        Args:
        None. The function assumes access to a globally defined threading.Event() named 'done_event'.
    
        Returns:
        None. This function returns nothing and is intended solely for side effects (console output).
        """
        dots = ["   ", ".  ", ".. ", "..."]  # List of dot states for animation.
        idx = 0  # Initialize index to cycle through dot states.
        print("Processing samples", end="")  # Initial print statement for processing message.
        while not done_event.is_set():  # Loop until the event is set signaling processing is complete.
            print(f"\rProcessing samples{dots[idx % len(dots)]}", end="", flush=True)  # Overwrite the previous line with new dot state.
            time.sleep(0.5)  # Pause for half a second before updating dot state.
            idx += 1  # Increment index to cycle to the next dot state.


    # Create and start the dot animation thread
    done_event = threading.Event()
    thread = threading.Thread(target=animate_dots)
    thread.start()

    try:
        
        # Define columns of interest
        columns_of_interest = ['taxonID', 'parentNameUsageID', 'acceptedNameUsageID', 'canonicalName', 'taxonRank', 'taxonomicStatus', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        gbif_full = pd.read_csv(gbif_path_file, sep="\t", usecols=columns_of_interest, on_bad_lines='skip', low_memory=False)
    
        # Filter the DataFrame
        gbif_subset = gbif_full.query("taxonomicStatus == 'accepted' & taxonRank != 'unranked'").fillna('').drop_duplicates(subset=gbif_full.columns[1:], keep='first')
    
        # Process the taxonomy data
        gbif_subset['gbif_taxonomy'] = gbif_subset.apply(create_gbif_taxonomy, axis=1)
        gbif_subset['gbif_taxonomy'] = gbif_subset['gbif_taxonomy'].str.replace(r'(\W)\1+', r'\1', regex=True)
    
        # Remove trailing and leading semicolons
        gbif_subset['gbif_taxonomy'] = gbif_subset['gbif_taxonomy'].str.rstrip(';').str.lstrip(';')
    
        # Remove rows with only semicolons
        gbif_subset.loc[gbif_subset['gbif_taxonomy'] == ';', 'gbif_taxonomy'] = ''
        gbif_subset = gbif_subset.drop_duplicates(subset="gbif_taxonomy")
    
        # Handle missing parent IDs
        #gbif_subset['parentNameUsageID'] = gbif_subset['parentNameUsageID'].replace('', np.nan).fillna(-1).astype(int)
        gbif_subset['parentNameUsageID'] = np.where(gbif_subset['parentNameUsageID'] == '', -1, gbif_subset['parentNameUsageID']).astype(int)
    
    
        parents_dict = dict(zip(gbif_subset['taxonID'], gbif_subset['parentNameUsageID']))
        gbif_subset['gbif_taxonomy_ids'] = gbif_subset.apply(lambda x: find_all_parents(x['taxonID'], parents_dict), axis=1)

    finally:
        done_event.set()
        thread.join()
        print("\rProcessing samples...")
        print("Done.")
    
    return gbif_subset, gbif_full


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
    tsv_path = os.path.join(gbif_output_folder, 'Taxon.tsv')

    # Check if the taxonomy data file already exists to avoid re-downloading
    if os.path.exists(tsv_path):
        print("GBIF backbone taxonomy data already downloaded.")
    
    else:    
    
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

    gbif_subset, gbif_full = load_gbif_samples(tsv_path)

    return gbif_subset, gbif_full





def prepare_ncbi_strings(row):
    """
    Prepare NCBI taxonomy strings.

    Args:
    row (pd.Series): A row from the NCBI DataFrame.

    Returns:
    str: A cleaned taxonomy string.
    """
    parts = row['ncbi_target_string'].split(';')
    
    if row['ncbi_rank'] in ['species', 'subspecies', 'strain']:
        new_string = ';'.join(parts[1:-1]) + ';' + row['ncbi_canonicalName']
    else:
        new_string = ';'.join(parts[1:-1])
    
    return new_string.lower()

def remove_extra_separators(s):
    """
    Remove extra semicolons from a string.

    Args:
    s (str): The string to process.

    Returns:
    str: The string with extra semicolons removed.
    """
    return re.sub(r';+', ';', s)


def download_ncbi_taxonomy(output_folder=None):

    """
    Download, extract, and process NCBI taxonomy data.

    This function sets up the necessary directories, downloads the taxonomy dump from NCBI,
    extracts the contents, and processes the data to create structured pandas DataFrames 
    with lineage information.

    Args:
    output_folder (str, optional): The directory to save NCBI output. Defaults to the current working directory.

    Returns:
    tuple: A tuple containing two pandas DataFrames:
        - ncbi_subset: DataFrame with unique target strings.
        - ncbi_full: Full DataFrame with all taxonomy information.
    """
    
    # Set the default output folder to the current working directory if none is provided
    if output_folder is None:
        output_folder = os.getcwd()

    # Create a new folder for storing the NCBI taxonomic data
    NCBI_output_folder = os.path.join(output_folder, "NCBI_output")
    os.makedirs(NCBI_output_folder, exist_ok=True)

    # Define the URL for the NCBI taxonomic database dump
    url = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdmp.zip"
    tf = os.path.join(NCBI_output_folder, "taxdmp.zip")

    # Check if the taxonomy data file already exists to avoid re-downloading
    if os.path.exists(tf):
        print("NCBI taxonomy data already downloaded.")
    
    else:
        # Download the zip file with a progress bar
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading NCBI Taxonomic Data") as pbar:
            def report_hook(blocknum, blocksize, totalsize):
                pbar.update(blocknum * blocksize - pbar.n)
            urllib.request.urlretrieve(url, tf, reporthook=report_hook)
    
        # Decompress the downloaded zip file
        with zipfile.ZipFile(tf, "r") as zip_ref:
            zip_ref.extractall(NCBI_output_folder)
    
        # Check if the file was downloaded successfully
        if os.path.exists(tf):
            print("NCBI taxonomy has been downloaded successfully.")

    def animate_dots():
        """
        Animate a sequence of dots on the console to indicate processing activity.
            
        Args:
        None. The function assumes access to a globally defined threading.Event() named 'done_event'.
    
        Returns:
        None. This function returns nothing and is intended solely for side effects (console output).
        """
        dots = ["   ", ".  ", ".. ", "..."]  # List of dot states for animation.
        idx = 0  # Initialize index to cycle through dot states.
        print("Processing samples", end="")  # Initial print statement for processing message.
        while not done_event.is_set():  # Loop until the event is set signaling processing is complete.
            print(f"\rProcessing samples{dots[idx % len(dots)]}", end="", flush=True)  # Overwrite the previous line with new dot state.
            time.sleep(0.5)  # Pause for half a second before updating dot state.
            idx += 1  # Increment index to cycle to the next dot state.


    # Create and start the dot animation thread
    done_event = threading.Event()
    thread = threading.Thread(target=animate_dots)
    thread.start()

    try:
        # Read the nodes dump file
        nodes_path = os.path.join(NCBI_output_folder, 'nodes.dmp')
        names_path = os.path.join(NCBI_output_folder, 'names.dmp')

        # Read the nodes file using a specific field separator and specifying no header
        nodes_df = pd.read_csv(
            nodes_path, 
            sep=r'\t\|\t',
            header=None,
            usecols=range(13),
            names=[
                'ncbi_id', 'parent_tax_id', 'rank', 'embl_code', 'division_id', 
                'inherited_div_flag', 'genetic_code_id', 'inherited_GC_flag', 
                'mitochondrial_genetic_code_id', 'inherited_MGC_flag', 
                'GenBank_hidden_flag', 'hidden_subtree_root_flag', 'comments'
            ],
            dtype=str,
            engine='python'
        ).replace(r'\t\|$', '', regex=True)

        # Read the names dump file
        names_df = pd.read_csv(
            names_path, 
            sep=r'\t\|\t',
            header=None,
            usecols=range(4),
            names=['ncbi_id', 'name_txt', 'unique_name', 'name_class'],
            dtype=str,
            engine='python'
        ).replace(r'\t\|$', '', regex=True)

        # Filter to include only scientific names
        scientific_names_df = names_df[names_df['name_class'] == 'scientific name']
        
        # Map NCBI IDs to parent tax IDs and ranks
        name_map = pd.Series(scientific_names_df['name_txt'].values, index=scientific_names_df['ncbi_id']).to_dict()
        
        # Add columns for canonical name and rank using the maps
        parent_map = pd.Series(nodes_df['parent_tax_id'].values, index=nodes_df['ncbi_id']).to_dict()
        rank_map = pd.Series(nodes_df['rank'].values, index=nodes_df['ncbi_id']).to_dict()
        
        # Aggiungere le colonne per il nome scientifico e il rango diretti
        nodes_df['ncbi_canonicalName'] = nodes_df['ncbi_id'].apply(lambda x: name_map.get(x, ''))
        nodes_df['ncbi_rank'] = nodes_df['ncbi_id'].apply(lambda x: rank_map.get(x, ''))
        
        # Function to compute the entire lineage, excluding the root
        def get_lineage(ncbi_id, map_dict):
            lineage = []
            while ncbi_id in map_dict and ncbi_id != '1': 
                lineage.append(ncbi_id)
                ncbi_id = map_dict[ncbi_id]
            return lineage[::-1]
        
        # Build lineages
        nodes_df['ncbi_lineage_ids'] = nodes_df['ncbi_id'].apply(lambda x: get_lineage(x, parent_map))
        nodes_df['ncbi_lineage_names'] = nodes_df['ncbi_lineage_ids'].apply(lambda ids: [name_map.get(id, '') for id in ids])
        nodes_df['ncbi_lineage_ranks'] = nodes_df['ncbi_lineage_ids'].apply(lambda ids: [rank_map.get(id, '') for id in ids])
        
        # Convert list of lineage information to semicolon-separated strings
        nodes_df['ncbi_lineage_names'] = nodes_df['ncbi_lineage_names'].apply(lambda names: ';'.join(names))
        nodes_df['ncbi_lineage_ids'] = nodes_df['ncbi_lineage_ids'].apply(lambda ids: ';'.join(ids))
        nodes_df['ncbi_lineage_ranks'] = nodes_df['ncbi_lineage_ranks'].apply(lambda ranks: ';'.join(ranks))
        
        # Define target ranks and build target strings based on ranks
        target_ranks = {'superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'}
        
        def build_target_string(names, ranks):
            names = names.split(';')
            ranks = ranks.split(';')
            return ';'.join(name for name, rank in zip(names, ranks) if rank in target_ranks).lower()
        
        nodes_df['ncbi_target_string'] = nodes_df.apply(
            lambda row: build_target_string(row['ncbi_lineage_names'], row['ncbi_lineage_ranks']), axis=1)
        
        # Prepare final subsets for return
        ncbi_full = nodes_df[['ncbi_id', 'ncbi_lineage_names', 'ncbi_lineage_ids', 'ncbi_canonicalName', 'ncbi_rank', 'ncbi_lineage_ranks', 'ncbi_target_string']]
        ncbi_subset = ncbi_full.copy()
        ncbi_subset['ncbi_target_string'] = ncbi_subset.apply(prepare_ncbi_strings, axis=1)
        ncbi_subset["ncbi_target_string"] = ncbi_subset["ncbi_target_string"].apply(remove_extra_separators).str.strip(';')
        ncbi_subset = ncbi_subset.drop_duplicates(subset="ncbi_target_string")
    
    finally:
        done_event.set()
        thread.join()
        print("\rProcessing samples...")
        print("Done.")

    return ncbi_subset, ncbi_full
