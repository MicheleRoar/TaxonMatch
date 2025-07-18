import os
import re
import time
import zipfile
import logging
import requests
import threading
import numpy as np
import pandas as pd
import urllib.request
from tqdm import tqdm
from bs4 import BeautifulSoup
from importlib import resources
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

from taxonmatch.analysis_utils import get_gbif_synonyms
from taxonmatch.analysis_utils import get_ncbi_synonyms
from taxonmatch.loader import save_gbif_dictionary
from taxonmatch.loader import save_ncbi_dictionary



def create_gbif_taxonomy(row):
    """
    Create a taxonomy string for GBIF data, handling special cases and avoiding redundancy.

    Args:
    row (pd.Series): A row from the GBIF DataFrame.

    Returns:
    str: A semicolon-separated string of taxonomy levels.
    """
    # List of taxonomic levels in hierarchical order
    taxonomy_levels = ['phylum', 'class', 'order', 'family', 'genus']

    # Build the base taxonomy string
    base_taxonomy = ";".join([str(row[level]).lower() for level in taxonomy_levels if pd.notna(row[level])])

    # Check if canonicalName is already included in the base taxonomy
    if pd.notna(row['canonicalName']):
        canonical_lower = str(row['canonicalName']).lower()
        if canonical_lower in base_taxonomy.split(";"):
            return base_taxonomy  # Avoid adding duplicate canonicalName

    # Case for subspecies, variety, and form (all treated as subspecies)
    if row['taxonRank'] in ["subspecies", "variety", "form"]:
        if pd.notna(row['canonicalName']):
            species = " ".join(row['canonicalName'].split(" ")[:2]).lower()  # Extract species name
            return f"{base_taxonomy};{species};{row['canonicalName'].lower()}"

    # Case for species
    if row['taxonRank'] == 'species':
        return f"{base_taxonomy};{str(row['canonicalName']).lower()}" if pd.notna(row['canonicalName']) else base_taxonomy

    # General case: Add canonicalName if not redundant
    if pd.notna(row['canonicalName']):
        return f"{base_taxonomy};{str(row['canonicalName']).lower()}"

    return base_taxonomy


def clean_taxonomy_dataframe(df):
    """
    Filters a taxonomy DataFrame to ensure consistency between `canonicalName` and `genus`.
    Specifically, for rows where `taxonRank` is "species", "subspecies", "variety", or "form",
    it removes rows where the first part of `canonicalName` does not match `genus`.

    ###### Issues: Name parent mismatch #######

    Args:
        df (pd.DataFrame): Input taxonomy DataFrame.

    Returns:
        pd.DataFrame: Cleaned taxonomy DataFrame.
    """
    # Taxon ranks to filter
    taxa_to_check = ["species", "subspecies", "variety", "form"]

    # Filter rows where taxonRank is in the specified categories
    df_filtered = df[df["taxonRank"].isin(taxa_to_check)]

    # Keep only rows where the first part of canonicalName matches genus
    df_filtered = df_filtered[
        df_filtered.apply(
            lambda row: str(row["canonicalName"]).split(" ")[0] == str(row["genus"]), axis=1
        )
    ]

    # Combine with rows that are not in the specified categories
    df_final = pd.concat(
        [
            df[~df["taxonRank"].isin(taxa_to_check)],
            df_filtered,
        ]
    ).reset_index(drop=True)

    return df_final


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

def load_gbif_samples(gbif_path_file, source = None):
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
        columns_of_interest = ['taxonID', 'datasetID' ,'parentNameUsageID', 'acceptedNameUsageID', 'canonicalName', 'taxonRank', 'scientificNameAuthorship', 'taxonomicStatus', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        gbif_full = pd.read_csv(gbif_path_file, sep="\t", usecols=columns_of_interest, on_bad_lines='skip', low_memory=False)

        gbif_full['gbif_taxonomy'] = gbif_full.apply(create_gbif_taxonomy, axis=1)

        if source:
            gbif_full_ = gbif_full[gbif_full['datasetID'] == source]
            list_of_synonyms = list(set(gbif_full_[gbif_full_.taxonomicStatus == "synonym"].acceptedNameUsageID.dropna()))
            df_synonyms = gbif_full[gbif_full.taxonID.isin(list_of_synonyms)]
            #gbif_full = gbif_full_.copy(deep=True)

        if source:
            # Filter the DataFrame
            gbif_subset = gbif_full_.query("taxonomicStatus == 'accepted' & taxonRank != 'unranked'").fillna('').drop_duplicates(subset=gbif_full.columns[1:], keep='first')
            gbif_subset = pd.concat([gbif_subset, df_synonyms], ignore_index=True)
        else:
            gbif_subset = gbif_full.query("taxonomicStatus == 'accepted' & taxonRank != 'unranked'").fillna('').drop_duplicates(subset=gbif_full.columns[1:], keep='first')


        gbif_subset_cleaned = clean_taxonomy_dataframe(gbif_subset)

        # Process the taxonomy data
        gbif_subset_cleaned['gbif_taxonomy'] = gbif_subset_cleaned['gbif_taxonomy'].str.replace(r'(\W)\1+', r'\1', regex=True)

        # Remove trailing and leading semicolons
        gbif_subset_cleaned['gbif_taxonomy'] = gbif_subset_cleaned['gbif_taxonomy'].str.rstrip(';').str.lstrip(';')

        # Remove rows with only semicolons
        gbif_subset_cleaned.loc[gbif_subset_cleaned['gbif_taxonomy'] == ';', 'gbif_taxonomy'] = ''
        gbif_subset_cleaned = gbif_subset_cleaned.drop_duplicates(subset="gbif_taxonomy")

        # Handle missing parent IDs
        #gbif_subset['parentNameUsageID'] = gbif_subset['parentNameUsageID'].replace('', np.nan).fillna(-1).astype(int)
        gbif_subset_cleaned['parentNameUsageID'] = np.where(gbif_subset_cleaned['parentNameUsageID'] == '', -1, gbif_subset_cleaned['parentNameUsageID']).astype(int)

        if source:
            parents_dict = dict(zip(gbif_full['taxonID'].fillna(-1).astype(int), gbif_full['parentNameUsageID'].fillna(-1).astype(int)))
        else:   
            parents_dict = dict(zip(gbif_subset_cleaned['taxonID'], gbif_subset_cleaned['parentNameUsageID']))
            
        gbif_subset_cleaned['gbif_taxonomy_ids'] = gbif_subset_cleaned.apply(lambda x: find_all_parents(x['taxonID'], parents_dict), axis=1)

        # Remove rows where the number of elements in gbif_taxonomy and gbif_taxonomy_ids do not match
        gbif_subset_cleaned = gbif_subset_cleaned[
        gbif_subset_cleaned.apply(lambda x: len(str(x['gbif_taxonomy']).split(';')) == len(str(x['gbif_taxonomy_ids']).split(';')), axis=1)]


        #Download synonym dictionary
        dictionary_path = resources.files('taxonmatch.files.dictionaries') / 'gbif_dictionaries.pkl'
        # If the dictionary file doesn't exist, generate and save it
        if not os.path.exists(dictionary_path):
            #print("gbif_dictionaries.pkl not found. Generating from dataset...")
            gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids = get_gbif_synonyms((gbif_subset, gbif_full))
            save_gbif_dictionary(gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids)

    finally:
        done_event.set()
        thread.join()
        print("\rProcessing samples...")
        print("Done.")
    
    if source:
        return gbif_subset_cleaned, gbif_full_
    else:
        return gbif_subset_cleaned, gbif_full


def download_gbif_taxonomy(output_folder=None, source = None):
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

    gbif_subset, gbif_full = load_gbif_samples(tsv_path, source)

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



def generate_cleaned_ncbi_string(row, target_ranks):
    """
    Generate a cleaned NCBI taxonomy string with canonical name if applicable,
    and handle specific ranks like species, subspecies, and strain.

    Args:
    row (pd.Series): A row from the NCBI DataFrame.
    target_ranks (set): The set of target ranks to include in the string.

    Returns:
    str: A cleaned taxonomy string.
    """
    # Split lineage names and ranks
    names = row['ncbi_lineage_names'].split(';')
    ranks = row['ncbi_lineage_ranks'].split(';')
    
    # Build the target string based on specified ranks
    target_string_parts = [name for name, rank in zip(names, ranks) if rank in target_ranks]

    # Handle empty target_string_parts case
    if not target_string_parts:
        return row['ncbi_canonicalName'].lower()

    # Handle special case for species, subspecies, strain
    if row['ncbi_rank'] in ['species', 'subspecies', 'strain']:
        if row['ncbi_canonicalName'].lower() != target_string_parts[-1].lower():
            new_string = ';'.join(target_string_parts) + ';' + row['ncbi_canonicalName']
        else:
            new_string = ';'.join(target_string_parts)
    else:
        if row['ncbi_canonicalName'].lower() != target_string_parts[-1].lower():
            new_string = ';'.join(target_string_parts) + ';' + row['ncbi_canonicalName']
        else:
            new_string = ';'.join(target_string_parts)
    
    return new_string.lower()



def download_ncbi_taxonomy(output_folder=None, source=None):

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
        
        # Add columns for direct scientific name and rank
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
        target_ranks = {'phylum', 'class', 'order', 'family', 'genus', 'species', 'subspecies'}

        nodes_df['ncbi_target_string'] = nodes_df.apply(
            lambda row: generate_cleaned_ncbi_string(row, target_ranks), axis=1
        )   


        # Prepare final subsets for return
        ncbi_full = nodes_df[['ncbi_id', 'ncbi_lineage_names', 'ncbi_lineage_ids', 'ncbi_canonicalName', 'ncbi_rank', 'ncbi_lineage_ranks', 'ncbi_target_string']]
        ncbi_subset = ncbi_full.copy()
        ncbi_filtered = ncbi_subset[ncbi_subset["ncbi_rank"].isin(target_ranks)].copy()

       
        dictionary_path = resources.files('taxonmatch.files.dictionaries') / 'ncbi_dictionaries.pkl'

        # If the dictionary file doesn't exist, generate and save it
        if not os.path.exists(dictionary_path):
            #print("ncbi_dictionaries.pkl not found. Generating from names.dmp...")
            ncbi_synonyms_names, ncbi_synonyms_ids = get_ncbi_synonyms(str(names_path))
            save_ncbi_dictionary(ncbi_synonyms_names, ncbi_synonyms_ids)
    
    finally:
        done_event.set()
        thread.join()
        print("\rProcessing samples...")
        print("Done.")

    if source == "a3cat":
        return load_a3cat_dataframe(ncbi_filtered), ncbi_full
    if source == "beebiome":
        return get_bees_from_beebiome_dataframe(ncbi_filtered), ncbi_full         
    else:
        return ncbi_filtered, ncbi_full



# Function to create a session with retries
def create_session_with_retries(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=0.3,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Function to get the IUCN status
def fetch_iucn_status(taxon_id, session):
    url = f"https://api.gbif.org/v1/species/{taxon_id}/iucnRedListCategory"
    try:
        response = session.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('category', 'UNKNOWN')
        else:
            return 'UNKNOWN'
    except requests.ConnectionError:
        return 'UNKNOWN'

# Main function to add the IUCN column
def add_iucn_status_column(df, id_column='taxonID', delay=0.1, max_workers=20):
    taxon_ids = df[id_column].tolist()
    iucn_status_dict = {}
    session = create_session_with_retries()

    # Fetch IUCN statuses
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(fetch_iucn_status, taxon_id, session): taxon_id for taxon_id in taxon_ids}
        total = len(taxon_ids)
        completed = 0
        for future in as_completed(future_to_id):
            taxon_id = future_to_id[future]
            status = future.result()
            iucn_status_dict[taxon_id] = status
            completed += 1
            progress = (completed / total) * 100
            print(f"\rProgress: {progress:.2f}%", end="")
            time.sleep(delay)

    # Map IUCN status to original DataFrame
    df_ = df.copy()
    df_['iucnRedListCategory'] = df_[id_column].map(lambda x: iucn_status_dict.get(x, 'UNKNOWN'))
    print()  # Move to the next line after completion
    return df_



def load_a3cat_dataframe(ncbi_filtered):
    url = 'https://a3cat.unil.ch'
    
    # Make the HTTP request to the website
    response = requests.get(url)
    response.raise_for_status()  

    # Parse HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the item with the specified ID
    version_tag = soup.find(id="header-version")
    
    if version_tag:
        # Extract the version
        version = version_tag.text.strip()
        # Extract date from version
        date = version.split('v.')[1]
        # Generate file link
        download_link = f"{url}/data/a3cat/{date}.tsv"
        
        # Download the file and load the data into a pandas DataFrame
        df = pd.read_csv(download_link, sep='\t')
        df['TaxId'] = df['TaxId'].astype(int)
        ncbi_filtered = ncbi_filtered.copy()
        ncbi_filtered['ncbi_id'] = ncbi_filtered['ncbi_id'].astype(int)
        a3cat = pd.merge(ncbi_filtered, df, left_on='ncbi_id', right_on='TaxId', how='inner')

        print(f"a3cat {version} downloaded")
        return a3cat
    else:
        print("Last version not found, please download the dataset manually")
        return None



def load_inat_samples(inat_path_file):
    """
    Load iNaturalist samples from a file and process them.

    Args:
    inat_path_file (str): Path to the iNaturalist taxa CSV file.

    Returns:
    DataFrame: Processed DataFrame of iNaturalist data filtered for Arthropoda.
    """

    def animate_dots():
        """
        Animate a sequence of dots on the console to indicate processing activity.
        """
        dots = ["   ", ".  ", ".. ", "..."]
        idx = 0
        print("Processing samples", end="")
        while not done_event.is_set():
            print(f"\rProcessing samples{dots[idx % len(dots)]}", end="", flush=True)
            time.sleep(0.5)
            idx += 1

    # Start animated dots
    done_event = threading.Event()
    thread = threading.Thread(target=animate_dots)
    thread.start()

    try:
        # Load and filter data
        columns_of_interest = [
            "id", "parentNameUsageID", "kingdom", "phylum", "class", "order",
            "family", "genus", "specificEpithet", "infraspecificEpithet",
            "modified", "scientificName", "taxonRank"
        ]

        inat_full = pd.read_csv(inat_path_file, usecols=columns_of_interest, low_memory=False)
        inat_full.rename(columns={"scientificName": "canonicalName"}, inplace=True)
        inat_full['inat_taxonomy'] = inat_full.apply(create_gbif_taxonomy, axis=1)

        combined = inat_full.copy()
        combined.columns = ['inat_taxon_id', 'inat_parentNameUsageID', 'kingdom', 'phylum',
               'class', 'order', 'family', 'genus', 'specificEpithet',
               'infraspecificEpithet', 'inat_modified', 'inat_canonical_name', 'inat_taxonRank',
               'inat_taxonomy']

    finally:
        done_event.set()
        thread.join()
        print("\rProcessing samples...")
        print("Done.")
    
    return combined




def download_inat_taxonomy(output_folder=None):
    """
    Download the iNaturalist taxonomy dataset, extract the relevant CSV, and process it.

    Args:
    output_folder (str, optional): The folder where the dataset will be downloaded.
                                   Defaults to the current working directory.

    Returns:
    DataFrame: Processed DataFrame containing Arthropoda taxa.
    """


    if output_folder is None:
        output_folder = os.getcwd()

    inat_output_folder = os.path.join(output_folder, 'iNaturalist_output')
    tsv_path = os.path.join(inat_output_folder, 'taxa.csv')

    if os.path.exists(tsv_path):
        print("iNaturalist taxonomy data already downloaded.")
    else:
        os.makedirs(inat_output_folder, exist_ok=True)

        url = "https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca.zip"
        filename = os.path.join(output_folder, "inaturalist_taxonomy.zip")


        # Download with progress bar
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading iNaturalist Taxonomic Data") as pbar:
            def report_hook(blocknum, blocksize, totalsize):
                pbar.update(blocknum * blocksize - pbar.n)
            urllib.request.urlretrieve(url, filename, reporthook=report_hook)

        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extract('taxa.csv', inat_output_folder)

        if os.path.exists(tsv_path):
            os.remove(filename)
        else:
            raise RuntimeError("Error saving taxa.csv. Consider raising an issue on Github.")



    return load_inat_samples(tsv_path)



def select_inat_clade(inat_dataset, selected_string):
    """
    Filters a dataset based on a selected taxonomic string and modifies the 'inat_taxonomy' column
    to include only the information from the selected string onward.

    Args:
        inat_dataset (DataFrame): Dataset containing a column 'inat_taxonomy'.
        selected_string (str): Taxonomic string (e.g., family, genus, order, etc.) to filter and modify.

    Returns:
        DataFrame: A filtered dataset with the 'inat_taxonomy' column modified.

    Raises:
        ValueError: If the selected string is not found as an exact match in the 'inat_taxonomy' column.
    """

    # Function to modify the taxonomy string
    def modify_inat_taxonomy(taxonomy, selected_string):
        """
        Truncates the taxonomy string to start from the selected string onward.

        Args:
            taxonomy (str): Full taxonomy string.
            selected_string (str): Taxonomic string to truncate from.

        Returns:
            str: Modified taxonomy string starting from the selected string, or the original string if not found.
        """
        parts = taxonomy.split(';')
        if selected_string.lower() in [p.lower() for p in parts]:
            index = [p.lower() for p in parts].index(selected_string.lower())
            return ';'.join(parts[index:])  # Keep everything from the selected string onward
        return taxonomy  # Return the original string if not found

    # Filter the dataset where 'inat_taxonomy' contains the exact selected string
    def contains_exact_taxon(taxonomy, selected_string):
        parts = taxonomy.split(';')
        return selected_string.lower() in [p.lower() for p in parts]

    # Apply filtering for exact matches
    filtered_dataset = inat_dataset[
        inat_dataset['inat_taxonomy'].apply(lambda x: contains_exact_taxon(x, selected_string))
    ].copy()

    # Raise an error if the selected string is not found in any row
    if filtered_dataset.empty:
        raise ValueError(
            f"The selected string '{selected_string}' was not found as an exact match in the 'inat_taxonomy' column."
        )

    # Modify the 'inat_taxonomy' column in the filtered dataset
    filtered_dataset['inat_taxonomy'] = filtered_dataset['inat_taxonomy'].apply(
        lambda x: modify_inat_taxonomy(x, selected_string)
    )

    filtered_dataset_with_ids = build_taxonomy_ids_names_ranks(filtered_dataset)

    return filtered_dataset_with_ids


def build_taxonomy_ids_names_ranks(df):
    """
    Constructs taxonomic lineage info: IDs, names, and ranks.
    Uses `inat_taxonomy` as reference to determine truncation root (if pre-filtered).

    Adds 3 new columns:
        - inat_taxonomy_ids
        - inat_lineage_names
        - inat_lineage_ranks
    """

    # 1. Build parent/name/rank dictionaries
    id_to_parent = {}
    id_to_name = {}
    id_to_rank = {}

    for row in df.itertuples(index=False):
        try:
            taxon_id = int(row.inat_taxon_id)

            # Parent ID
            if pd.notna(row.inat_parentNameUsageID):
                parent_id_str = str(row.inat_parentNameUsageID).strip().split('/')[-1]
                if parent_id_str.isdigit():
                    id_to_parent[taxon_id] = int(parent_id_str)

            # Canonical name + rank
            if pd.notna(row.inat_canonical_name):
                id_to_name[taxon_id] = str(row.inat_canonical_name).strip().lower()
            if pd.notna(row.inat_taxonRank):
                id_to_rank[taxon_id] = str(row.inat_taxonRank).strip()
        except:
            continue

    # 2. Trace full lineage per row, then truncate to match inat_taxonomy
    ids_out, names_out, ranks_out = [], [], []

    for row in df.itertuples(index=False):
        if pd.isna(row.inat_taxon_id):
            ids_out.append("")
            names_out.append("")
            ranks_out.append("")
            continue

        lineage_ids = []
        lineage_names = []
        lineage_ranks = []

        tid = int(row.inat_taxon_id)
        seen = set()

        # Step A: trace full lineage up to root
        while tid and tid not in seen:
            lineage_ids.insert(0, tid)
            lineage_names.insert(0, id_to_name.get(tid, '').strip().lower())
            lineage_ranks.insert(0, id_to_rank.get(tid, '').strip())

            seen.add(tid)
            tid = id_to_parent.get(tid)
            if tid is None or tid == 1:
                break

        # Step B: truncate using inat_taxonomy
        if pd.notna(row.inat_taxonomy):
            ref_names = [x.strip().lower() for x in str(row.inat_taxonomy).split(';') if x.strip()]
            if ref_names:
                try:
                    start_idx = lineage_names.index(ref_names[0])
                    lineage_ids = lineage_ids[start_idx:]
                    lineage_names = lineage_names[start_idx:]
                    lineage_ranks = lineage_ranks[start_idx:]
                except ValueError:
                    # If not found, keep all (fallback)
                    pass

        ids_out.append(';'.join(str(x) for x in lineage_ids))
        names_out.append(';'.join(lineage_names))
        ranks_out.append(';'.join(lineage_ranks))

    df['inat_taxonomy_ids'] = ids_out
    df['inat_lineage_names'] = names_out
    df['inat_lineage_ranks'] = ranks_out

    return df



def get_bees_from_beebiome_dataframe(ncbi_filtered):
    # API URL to extract data from
    url = "https://beebiome.org/beebiome/sample/all"
    
    # Make the request to the API
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Convert the response to JSON format
        data = response.json()
    
        # Extract all host IDs
        host_ids = [sample['host']['id'] for sample in data if 'host' in sample and 'id' in sample['host']]
    
        # Print host IDs
        beebiome_ids = (list(set(host_ids)))
        return ncbi_filtered[ncbi_filtered.ncbi_id.astype(int).isin(beebiome_ids)]
    else:
        print("Error in request:", response.status_code)


