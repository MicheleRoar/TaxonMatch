import re
import pandas as pd
import numpy as np
from importlib import resources
import pickle

def load_xgb_model():
    with resources.open_binary('taxonmatch.files.models', 'xgb_model.pkl') as file:
        model = pickle.load(file)
    return model

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

def load_ncbi_samples(ncbi_path_file):
    """
    Load NCBI samples from a file and process them.

    Args:
    ncbi_path_file (str): Path to the NCBI file.

    Returns:
    tuple: A tuple containing the processed subset and the full DataFrame of NCBI data.
    """
    # Read and process the NCBI file
    ncbi_full = pd.read_csv(ncbi_path_file, sep="\t", names=['ncbi_id', 'ncbi_lineage_names', 'ncbi_lineage_ids', 'ncbi_canonicalName', 'ncbi_rank', 'ncbi_lineage_ranks', 'ncbi_target_string'])
    ncbi_subset = ncbi_full.copy()
    ncbi_subset['ncbi_target_string'] = ncbi_subset.apply(prepare_ncbi_strings, axis=1)
    ncbi_subset["ncbi_target_string"] = ncbi_subset["ncbi_target_string"].apply(remove_extra_separators).str.strip(';')
    ncbi_subset = ncbi_subset.drop_duplicates(subset="ncbi_target_string")
    
    return ncbi_subset, ncbi_full


def load_xgb_model():
    with resources.open_binary('taxonmatch.files.models', 'xgb_model.pkl') as file:
        model = pickle.load(file)
    return model


def load_gbif_dictionary():
    with resources.open_binary('taxonmatch.files.dictionaries', 'gbif_dictionaries.pkl') as file:
        gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids = pickle.load(file)
    return gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids

def load_ncbi_dictionary():
    with resources.open_binary('taxonmatch.files.dictionaries', 'ncbi_dictionaries.pkl') as file:
        ncbi_synonyms_names, ncbi_synonyms_ids = pickle.load(file)
    return ncbi_synonyms_names, ncbi_synonyms_ids