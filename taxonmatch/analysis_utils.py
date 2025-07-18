
import os
import re
import pickle
import requests
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textdistance import levenshtein
from taxonmatch.loader import save_gbif_dictionary, save_ncbi_dictionary


def get_gbif_synonyms(gbif_dataset):
    """
    Extracts synonyms from the GBIF dataset and generates three dictionaries:

    1. A dictionary mapping acceptedNameUsageID to their synonyms (canonical names).
    2. A dictionary mapping the accepted canonical name to its corresponding synonyms.
    3. A dictionary mapping acceptedNameUsageID to lists of tuples (synonym name, synonym ID).
    
    Args:
    gbif_dataset (DataFrame): GBIF DataFrame.

    Returns:
    tuple: Three dictionaries (synonyms by name, synonyms by ID, synonyms ID → (name, ID)).
    """
    if not isinstance(gbif_dataset[1], pd.DataFrame):
        raise ValueError("gbif_dataset has to be a pandas DataFrame.")

    # Filter only the synonyms.
    df_gbif_synonyms = gbif_dataset[1][gbif_dataset[1]['taxonomicStatus'] == 'synonym']

    gbif_synonyms_ids = {}
    gbif_synonyms_names = {}
    gbif_synonyms_ids_to_ids = {}  

    # Map taxonID → canonicalName
    id_to_canonical = gbif_dataset[1].set_index('taxonID')['canonicalName'].to_dict()

    for _, row in df_gbif_synonyms.iterrows():
        accepted_id = row['acceptedNameUsageID']
        synonym_canonical_name = row['canonicalName']
        synonym_id = row['taxonID']  

        accepted_canonical_name = id_to_canonical.get(accepted_id)

        if pd.notna(synonym_canonical_name) and pd.notna(accepted_canonical_name):
            gbif_synonyms_ids.setdefault(accepted_id, set()).add(synonym_canonical_name)
            gbif_synonyms_names.setdefault(accepted_canonical_name, set()).add(synonym_canonical_name)
            gbif_synonyms_ids_to_ids.setdefault(accepted_id, set()).add((synonym_canonical_name, synonym_id))  

    # Converts sets into lists
    gbif_synonyms_ids = {key: list(value) for key, value in gbif_synonyms_ids.items()}
    gbif_synonyms_names = {key: list(value) for key, value in gbif_synonyms_names.items()}
    gbif_synonyms_ids_to_ids = {key: list(value) for key, value in gbif_synonyms_ids_to_ids.items()}  

    # Save dictionary
    save_gbif_dictionary(gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids)

    return gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids


def get_ncbi_synonyms(names_path):
    """
    Extracts synonyms from the NCBI names.dmp file and generates two dictionaries:

    A dictionary mapping the scientific name to its synonyms.
    A dictionary mapping the taxonomic ID to its synonyms.
    Args:
    names_path (str): Path to the names.dmp file.
    Returns:
    tuple: Two dictionaries (synonyms by name, synonyms by ID).
    """
    scientific_names = {}
    ncbi_synonyms_names = {}
    ncbi_synonyms_ids = {}

    # Identify scientific names
    with open(names_path, 'r') as names_file:
        for line in names_file:
            parts = line.strip().split('|')
            taxid = int(parts[0].strip())
            name = parts[1].strip().strip("'")
            name_type = parts[3].strip()

            if name_type == 'scientific name':
                scientific_names[taxid] = name
                ncbi_synonyms_names.setdefault(name, set())
                ncbi_synonyms_ids.setdefault(taxid, set())

    # Associate synonyms with IDs and names
    with open(names_path, 'r') as names_file:
        for line in names_file:
            parts = line.strip().split('|')
            taxid = int(parts[0].strip())
            name = parts[1].strip().strip("'")
            name_type = parts[3].strip()

            if name_type in {'acronym', 'blast name', 'common name', 'equivalent name', 'genbank acronym', 'genbank common name', 'synonym'}:
                if taxid in scientific_names:
                    scientific_name = scientific_names[taxid]
                    if name.lower() != scientific_name.lower():
                        ncbi_synonyms_names[scientific_name].add(name)
                        ncbi_synonyms_ids[taxid].add(name)

    # Convert sets into lists
    ncbi_synonyms_names = {key: list(value) for key, value in ncbi_synonyms_names.items()}
    ncbi_synonyms_ids = {key: list(value) for key, value in ncbi_synonyms_ids.items()}

    return ncbi_synonyms_names, ncbi_synonyms_ids


    # Save dictionaries
    save_ncbi_dictionary(ncbi_synonyms_names, ncbi_synonyms_ids)


def get_inconsistencies(gbif_dataset, ncbi_dataset):

    matched_df = gbif_dataset[0].merge(ncbi_dataset[0], left_on='canonicalName', right_on= 'ncbi_canonicalName', how='inner')
    double_cN = matched_df.groupby('canonicalName').filter(lambda x: len(set(x['kingdom'])) > 1)
    double_cN = double_cN[double_cN["taxonomicStatus"] == "accepted"]
    double_cN_filtered = double_cN[double_cN['gbif_taxonomy'].str.count(';') > 0]
    double_cN_filtered = double_cN_filtered[double_cN_filtered['ncbi_target_string'].str.count(';') > 0]
    double_cN_filtered = double_cN_filtered[["canonicalName", "taxonID", 'ncbi_id', "taxonRank", "ncbi_rank", "gbif_taxonomy", "ncbi_target_string"]].drop_duplicates()
    double_cN_filtered['distance'] = double_cN_filtered.apply(lambda x: levenshtein.distance(x["gbif_taxonomy"], x["ncbi_target_string"]), axis=1)
    false_matches = double_cN_filtered.query('distance > 40')
    false_matches.columns = ['canonicalName', 'gbif_id', 'ncbi_id', 'gbif_rank', 'ncbi_rank', 'gbif_taxonomy', 'ncbi_taxonomy', 'distance']
    df_inconsistencies = false_matches[false_matches['canonicalName'].apply(lambda x: len(x.split()) == 2)]
    
    return df_inconsistencies.drop(columns=["distance"])


def get_wordcloud(full_training_set):
    """
    Generates and displays a word cloud from a given dataset.

    Args:
    full_training_set (DataFrame): A pandas DataFrame containing the dataset.

    This function randomly samples 5000 entries from the provided dataset and concatenates
    the 'gbif_name' and 'ncbi_name' fields to create a large text string. A word cloud is
    then generated from this text and displayed.
    """
    texts = ''
    for index, item in full_training_set.sample(5000).iterrows():
        texts += ' ' + item['gbif_name'] + ' ' + item['ncbi_name']
    word_cloud = WordCloud(collocations=False, background_color='white').generate(texts)
    
    # Plot the WordCloud image
    plt.figure(figsize=(5, 10), facecolor=None)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()



def find_dataset_ids_by_name(name):
    url = "https://api.gbif.org/v1/dataset"
    params = {'q': name, 'limit': 10}

    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        
        if not results:
            print("No datasets found.")
            return []

        dataset_ids = []
        for dataset in results:
            print(f"Title: {dataset['title']}, ID: {dataset['key']}")
            dataset_ids.append(dataset['key'])

        #return dataset_ids
    else:
        print(f"Error during request: {response.status_code}")
        return []

def find_species_information(species, dataset):
    """
    Searches for a species in the dataset based on either 'canonicalName' or 'ncbi_canonicalName',
    depending on which column exists.

    Args:
        species (str): The species name to search for.
        dataset (pd.DataFrame): The dataset containing taxonomic information.

    Returns:
        pd.DataFrame | str: A filtered DataFrame with matching species or "String not found" message.
    """
    # Determine which column exists in the dataset
    possible_columns = ['canonicalName', 'ncbi_canonicalName']
    available_columns = [col for col in possible_columns if col in dataset[0].columns]

    if not available_columns:
        return "Error: No valid name column found in dataset"

    # Perform the filtering based on the existing column(s)
    result = dataset[0][dataset[0][available_columns].eq(species).any(axis=1)]

    return result if not result.empty else "String not found"


def clean_taxon_names(df, column_name):
    """
    Cleans taxonomic names and stores the cleaned version in a new column with '_cleaned' suffix.

    - Removes everything inside parentheses (including parentheses).
    - Removes taxonomic prefixes 'subsp.' and 'var.'.
    - Removes all words with an uppercase initial letter (except the first word).
    - Removes special characters, including '.', '&', ',', and "'".
    - Normalizes accented characters to ensure complete cleaning.
    - Removes any remaining author names after normalization.
    - Removes the word 'ex' (often used in taxonomic attributions).

    :param df: Pandas DataFrame containing the column with taxonomic names.
    :param column_name: Name of the column to clean.
    :return: DataFrame with an additional column containing cleaned taxon names.
    """
    df = df.copy()  # Prevent modifying the original DataFrame

    # Define the new column name
    new_column_name = f"{column_name}_cleaned"

    # Create a new column to store cleaned names
    df[new_column_name] = df[column_name].astype(str)

    # Remove everything inside parentheses (including parentheses)
    df[new_column_name] = df[new_column_name].str.replace(r"\s*\(.*?\)", "", regex=True)

    # Remove prefixes 'subsp.' and 'var.'
    df[new_column_name] = df[new_column_name].str.replace(r'\b(subsp|var)\b', '', regex=True)

    # Remove special characters (including ., &, ',', and apostrophe ')
    df[new_column_name] = df[new_column_name].str.replace(r"[.,&']", "", regex=True)

    # Normalize characters to handle accented letters
    df[new_column_name] = df[new_column_name].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8'))

    # Remove all words with an uppercase initial letter (except the first word)
    df[new_column_name] = df[new_column_name].apply(
        lambda x: ' '.join([w for i, w in enumerate(x.split()) if i == 0 or not re.match(r"^[A-Z][a-zA-Z-'’]*$", w)])
    )

    # Remove the word 'ex' (often used in taxonomic attributions)
    df[new_column_name] = df[new_column_name].str.replace(r'\bex\b', '', regex=True)

    # Remove multiple spaces generated by cleaning
    df[new_column_name] = df[new_column_name].str.replace(r'\s+', ' ', regex=True).str.strip()

    return df


def plot_conservation_statuses(df_with_iucn_status):
    
    # Define conservation statuses in order of increasing threat level
    conservation_status_order = [
        'Critically Endangered', 'Endangered', 'Vulnerable', 'Near Threatened', 
        'Data Deficient', 'Least Concern'
    ]
    
    # Define colors for each category (red → green gradient)
    status_colors = {
        'Critically Endangered': '#e41a1c',  # Dark red
        'Endangered': '#ff7f00',  # Orange
        'Vulnerable': '#e6ab02',  # Yellow
        'Near Threatened': '#377eb8',  # Blue
        'Data Deficient': '#984ea3',  # Purple
        'Least Concern': '#4daf4a'  # Green
    }
    
    # Normalize category names: replace underscores, title-case names
    df_with_iucn_status['iucnRedListCategory'] = df_with_iucn_status['iucnRedListCategory']\
        .str.replace('_', ' ')\
        .str.title()
    
    # Ensure all categories are correctly mapped
    df_with_iucn_status['iucnRedListCategory'] = df_with_iucn_status['iucnRedListCategory'].map(
        lambda x: x if x in conservation_status_order else None
    )
    
    # Remove rows with categories not in our defined order
    df_with_iucn_status = df_with_iucn_status.dropna(subset=['iucnRedListCategory'])
    
    # Count species per conservation status
    conservation_counts = df_with_iucn_status['iucnRedListCategory'].value_counts()
    
    # Reindex to ensure all categories are included in the correct order
    conservation_counts = conservation_counts.reindex(conservation_status_order, fill_value=0)
    
    # Generate the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(conservation_counts.index, conservation_counts.values, 
                   color=[status_colors[status] for status in conservation_counts.index], 
                   edgecolor='black')
    
    # Adjust the y-axis limit to prevent the tallest bar from touching the top
    plt.ylim(0, max(conservation_counts.values) * 1.15)  # Adds 15% extra space
    
    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(conservation_counts.values) * 0.02), 
                 str(int(yval)), ha='center', fontsize=9, fontweight='bold')
    
    # Set title and labels
    plt.title('Species Distribution by Conservation Status', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Species with Genome Available', fontsize=11)
    plt.xlabel('Conservation Status', fontsize=11)
    
    # Improve x-axis readability
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()



def fast_check_taxon_consistency(df):
    problems = {}

    print("⚡ Starting generic taxonomic consistency check\n")

    # Dynamically detect relevant columns
    id_cols = [col for col in df.columns if col.endswith('_taxon_id')]
    name_cols = [col for col in df.columns if col.endswith('_canonical_name')]

    print(f"🔹 ID columns: {id_cols}")
    print(f"🔹 Canonical name columns: {name_cols}\n")

    #1. Check for duplicate IDs within each column
    for col in id_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna().values
        unique, counts = np.unique(values, return_counts=True)
        dup_ids = unique[counts > 1].tolist()
        if dup_ids:
            print(f"Duplicates found in {col}: {dup_ids}")
            problems[f'duplicated_{col}'] = dup_ids
        else:
            print(f"No duplicates found in {col}")

    print("\n---")

    #2. Check for duplicate canonical names within each column
    for col in name_cols:
        values = df[col].dropna().astype(str).values
        unique, counts = np.unique(values, return_counts=True)
        dup_names = unique[counts > 1].tolist()
        if dup_names:
            print(f"Duplicates found in canonical names column {col}: {dup_names}")
            problems[f'duplicated_{col}'] = dup_names
        else:
            print(f"No duplicates found in canonical names column {col}")

    print("\n---")

    #3. Cross-column canonical name conflicts across different rows
    print("🔎 Checking for cross-column canonical name overlaps across different rows...\n")
    seen = {}
    cross_conflicts = set()

    for idx, row in df[name_cols].iterrows():
        for val in row.dropna().astype(str).values:
            if val in seen and seen[val] != idx:
                print(f"Name '{val}' appears in multiple rows ({seen[val]} and {idx}) across different columns")
                cross_conflicts.add(val)
            else:
                seen[val] = idx

    if cross_conflicts:
        problems['canonical_name_overlap_across_columns_different_rows'] = list(cross_conflicts)
    else:
        print("No overlapping canonical names across different rows")

    print("\n---")

    #4. Rows missing all canonical names (inat is optional)
    required_sources = ['ncbi', 'gbif']
    optional_sources = ['inat']
    all_name_cols = [f"{src}_canonical_name" for src in required_sources]
    if 'inat_canonical_name' in df.columns:
        all_name_cols.append('inat_canonical_name')

    condition = df[all_name_cols].isna().all(axis=1)
    missing_all_canonicals = df[condition]

    if not missing_all_canonicals.empty:
        print(f"Rows missing all canonical names: {len(missing_all_canonicals)}")
        problems['missing_all_canonical_names'] = missing_all_canonicals.index.tolist()
        display(missing_all_canonicals)
    else:
        print("All rows have at least one canonical name")

    print("\n---")

    #5. Taxon ID present but canonical name missing (inat optional)
    for source in ['gbif', 'ncbi', 'inaturalist']:
        id_col = f"{source}_taxon_id"
        name_col = f"{source}_canonical_name"
        if id_col in df.columns and name_col in df.columns:
            missing_name = df[df[id_col].notna() & df[name_col].isna()]
            if not missing_name.empty:
                print(f"❌ Rows with {id_col} present but missing {name_col}: {len(missing_name)}")
                problems[f'{source}_id_without_name'] = missing_name.index.tolist()
                display(missing_name)
            else:
                print(f"All rows with {id_col} have a corresponding {name_col}")

    print("\nConsistency check completed.\n")
    return problems


def get_dataset_from_species(species_name, output_folder=None):
    """
    Searches for a species in the GBIF Taxon.tsv file located in ./GBIF_output/
    and returns the dataset title and ID from which the species originates.

    Args:
        species_name (str): Exact species name (e.g., "Ristoria pliocaenica")
        output_folder (str or None): Base folder containing 'GBIF_output'. Defaults to current working directory.

    Returns:
        tuple: (datasetID, dataset title) if found, otherwise an error message string.
    """
    # Use current working directory if no output folder is provided
    if output_folder is None:
        output_folder = os.getcwd()

    # Construct the full path to the Taxon.tsv file
    gbif_output_folder = os.path.join(output_folder, 'GBIF_output')
    tsv_path = os.path.join(gbif_output_folder, 'Taxon.tsv')

    # Check if the file exists
    if not os.path.isfile(tsv_path):
        return f"File not found: {tsv_path}"

    # Load the Taxon.tsv file
    try:
        gbif_df = pd.read_csv(tsv_path, sep="\t", on_bad_lines='skip', low_memory=False)
    except Exception as e:
        return f"Error loading file: {e}"

    # Look for an exact match on the canonicalName field
    match = gbif_df[gbif_df['canonicalName'].fillna('') == species_name]

    if match.empty:
        return f"Species '{species_name}' not found in {tsv_path}"

    # Get the dataset ID associated with the matched species
    dataset_id = match['datasetID'].values[0]

    # Query the GBIF API to get dataset metadata
    url = f"https://api.gbif.org/v1/dataset/{dataset_id}"
    response = requests.get(url)

    if response.status_code != 200:
        return f"API error ({response.status_code}) for dataset {dataset_id}."

    # Extract and return the dataset title
    data = response.json()
    title = data.get("title", "Title not available")

    return dataset_id, title

