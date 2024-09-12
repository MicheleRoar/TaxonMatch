
import pandas as pd
import requests
import pickle
import os
from textdistance import levenshtein
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def get_gbif_synonyms(gbif_dataset):
    """
    Extracts synonyms from a GBIF dataset and returns three dictionaries:
    1. A dictionary mapping accepted IDs to their synonyms (canonical names).
    2. A dictionary mapping the canonical name of each accepted ID to the canonical names of all associated synonyms.
    3. A dictionary mapping accepted IDs to the IDs of their synonyms.
    
    Args:
    gbif_dataset (DataFrame): A pandas DataFrame containing GBIF data.
    
    Returns:
    tuple: Three dictionaries as described above.
    """
    if not isinstance(gbif_dataset[1], pd.DataFrame):
        raise ValueError("gbif_dataset must be a pandas DataFrame.")
    
    # Filter for synonyms
    df_gbif_synonyms = gbif_dataset[1][gbif_dataset[1]['taxonomicStatus'].str.contains('synonym', na=False)]
    
    gbif_synonyms_ids = {}
    gbif_synonyms_names = {}
    gbif_synonyms_ids_to_ids = {}  

    # Map each acceptedNameUsageID to its canonicalName
    id_to_canonical = gbif_dataset[1].set_index('taxonID')['canonicalName'].to_dict()

    for index, row in df_gbif_synonyms.iterrows():
        accepted_id = row['acceptedNameUsageID']
        synonym_canonical_name = row['canonicalName']
        synonym_id = row['taxonID']  
        
        # Obtain the canonicalName of the acceptedNameUsageID
        accepted_canonical_name = id_to_canonical.get(accepted_id)

        if not pd.isnull(synonym_canonical_name) and not pd.isnull(accepted_canonical_name):
            gbif_synonyms_ids.setdefault(accepted_id, set()).add(synonym_canonical_name)
            gbif_synonyms_names.setdefault(accepted_canonical_name, set()).add(synonym_canonical_name)
            gbif_synonyms_ids_to_ids.setdefault(accepted_id, set()).add(synonym_id)

    # Convert sets to lists for each key in the dictionaries
    gbif_synonyms_ids = {key: list(value) for key, value in gbif_synonyms_ids.items()}
    gbif_synonyms_names = {key: list(value) for key, value in gbif_synonyms_names.items()}
    gbif_synonyms_ids_to_ids = {key: list(value) for key, value in gbif_synonyms_ids_to_ids.items()}

    # Define the directory path and file name
    directory = './files/dictionaries'
    filename = 'gbif_dictionaries.pkl'
    file_path = os.path.join(directory, filename)

    # Ensure the directory exists; if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the dictionaries in a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump((gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids), f)

    return gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids


def get_ncbi_synonyms(names_path):
    scientific_names = {}  # Mappa ogni TaxID al suo nome scientifico
    ncbi_synonyms_names = {}  # Dizionario che mappa i nomi scientifici ai loro sinonimi e altri nomi
    ncbi_synonyms_ids = {}  # Dizionario che mappa gli ID ai loro sinonimi e altri nomi

    # Prima passata: identifica i nomi scientifici per ciascun TaxID
    with open(names_path, 'r') as names_file:
        for line in names_file:
            parts = line.strip().split('|')
            taxid = int(parts[0].strip())
            name = parts[1].strip().strip("'")
            name_type = parts[3].strip()

            if name_type == 'scientific name':
                scientific_names[taxid] = name
                if name not in ncbi_synonyms_names:
                    ncbi_synonyms_names[name] = []
                if taxid not in ncbi_synonyms_ids:  # Inizializza anche la lista per l'ID nel dizionario ncbi_synonyms_ids
                    ncbi_synonyms_ids[taxid] = []

    # Seconda passata: associa gli altri nomi ai nomi scientifici e agli ID corrispondenti
    with open(names_path, 'r') as names_file:
        for line in names_file:
            parts = line.strip().split('|')
            taxid = int(parts[0].strip())
            name = parts[1].strip().strip("'")
            name_type = parts[3].strip()

            if name_type in ['acronym', 'blast name', 'common name', 'equivalent name', 'genbank acronym', 'genbank common name', 'synonym']:
                if taxid in scientific_names:
                    scientific_name = scientific_names[taxid]
                    # Aggiunge il nome al dizionario ncbi_synonyms_names sotto il nome scientifico corrispondente
                    if name.lower() != scientific_name.lower():
                        ncbi_synonyms_names[scientific_name].append(name)
                    # Aggiunge il nome al dizionario ncbi_synonyms_ids sotto l'ID corrispondente
                    ncbi_synonyms_ids[taxid].append(name)

    # Definisci il percorso della directory e il nome del file
    directory = './files/dictionaries'
    filename = 'ncbi_dictionaries.pkl'
    file_path = os.path.join(directory, filename)

    # Assicurati che la directory esista; in caso contrario, creala
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open('./files/dictionaries/ncbi_dictionaries.pkl', 'wb') as f:
        pickle.dump((ncbi_synonyms_names, ncbi_synonyms_ids), f)

    return ncbi_synonyms_names, ncbi_synonyms_ids


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
    return false_matches


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



def find_dataset_id_by_name(name):
    # URL of the API for searching datasets in GBIF
    url = "https://api.gbif.org/v1/dataset"

    # Parameters for the simple text search
    params = {
        'q': name,  # Search by name
        'limit': 10  # Limit of returned results
    }

    # Make the request to the GBIF API
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])

        if not results:
            print("No datasets found.")
            return None

        # Print and return the ID of the first matching dataset
        for dataset in results:
            print(f"Title: {dataset['title']}, ID: {dataset['key']}")
            return dataset['key']
    else:
        print(f"Error during request: {response.status_code}")
        return None