import re
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from .model_training import tuple_engineer_features

def ngrams(string, n=4):

    """
    Generate n-grams from a given string. The process includes removing non-ascii characters, normalization, and cleaning.

    Args:
    string (str): The string to generate n-grams from.
    n (int): The number of characters in each n-gram.

    Returns:
    list: A list of n-grams generated from the string.
    """


    string = string.encode("ascii", errors="ignore").decode()
    string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()
    string = re.sub(' +',' ',string).strip()
    string = ' '+ string +' '
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]



def extract_after_arthropoda(clade, lineage_names):
    """
    Extracts the portion of the lineage string following the selected clade.
    
    Args:
    lineage_names (str): A semicolon-separated string of lineage names.

    Returns:
    str: A modified lineage string starting from the selected clade.
    """
    return clade.lower() + ";" + lineage_names.split(clade + ';', 1)[-1].lower()


def select_taxonomic_clade(clade, gbif_dataset, ncbi_dataset):

	"""
    Selects and processes entries from the GBIF and NCBI datasets that belong to a specified taxonomic clade.

    Args:
    clade (str): The name of the taxonomic clade to filter for.
    gbif_dataset (DataFrame): The GBIF dataset.
    ncbi_dataset (DataFrame): The NCBI dataset.

    Returns:
    tuple: Processed GBIF and NCBI datasets containing entries of the specified clade.
    """

    # Filter and extract rows of interest from the gbif_dataset DataFrame
    gbif_arthropoda = gbif_dataset[0][gbif_dataset[0].phylum == clade]

    # Filter and apply the extraction function to the DataFrame ncbi_arthropoda
    ncbi_arthropoda = ncbi_dataset[0][ncbi_dataset[0]["ncbi_target_string"].str.contains("Arthropoda".lower() + ";")]
    ncbi_arthropoda_ = ncbi_arthropoda.copy()
    ncbi_arthropoda_.loc[:, 'ncbi_lineage_names'] = ncbi_arthropoda_['ncbi_lineage_names'].apply(extract_after_arthropoda)

    # Filtering dataframe to exclude rows containing numbers in ncbi_lineage_names
    #ncbi_arthropoda_ = ncbi_arthropoda_[~ncbi_arthropoda_['ncbi_canonicalName'].str.contains(r'\d')]

    # Update ncbi_lineage_ids to match the number of elements in the new ncbi_lineage_names
    ncbi_arthropoda_['ncbi_lineage_ids'] = ncbi_arthropoda_.apply(lambda row: ';'.join(row['ncbi_lineage_ids'].split(';')[-len(row['ncbi_lineage_names'].split(';')):]), axis=1)

    ncbi_arthropoda = ncbi_arthropoda_.iloc[1:]
    return ncbi_arthropoda, gbif_arthropoda

def match_datasets(query_dataset, target_dataset, model, relevant_features, threshold):

	"""
    Matches datasets using a model and calculates the probability of correct matches.

    Args:
    query_dataset (DataFrame): The dataset to query.
    target_dataset (DataFrame): The dataset to match against.
    model: The machine learning model used for matching.
    relevant_features: Features relevant to the matching process.
    threshold (float): The threshold for determining a match.

    Returns:
    tuple: DataFrames of matched and unmatched entries.
    """

    target_dataset_2 = target_dataset[~target_dataset['ncbi_canonicalName'].str.contains(r'\d')]
    
    target = list(set(target_dataset_2.ncbi_target_string))
    query = list(set(query_dataset.gbif_taxonomy))
    
    vectorizer = TfidfVectorizer(analyzer=ngrams, lowercase=True)
    tfidf = vectorizer.fit_transform(target)

    discarded = []
    matches_df = []

    # Configura il vettorizzatore TF-IDF
    vectorizer = TfidfVectorizer(analyzer=ngrams, lowercase=True)
    tfidf = vectorizer.fit_transform(target)

    # Configura il modello NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto', n_jobs=-1, metric='cosine').fit(tfidf)

    # Calcola le distanze e gli indici dei vicini
    distances, indices = nbrs.kneighbors(vectorizer.transform(query))
    distances = np.round(distances, 2)

    n_samples, n_neighbors = distances.shape

    expanded_distances = distances.flatten()
    expanded_query = np.repeat(query, n_neighbors)
    expanded_target = np.array(target)[indices.flatten()]

    matches = np.column_stack((expanded_distances, expanded_query, expanded_target))
    matches_ = matches[:, [1, 2, 0]]

    features = tuple_engineer_features(matches_, relevant_features)
    filtered_features = [feature for feature in features if float(feature['score']) < 0.30]

    y_pred = model.predict_proba(extract_values_from_dict_list(filtered_features))[:, 1]
    add_predictions_to_features(filtered_features, y_pred)

    best_match_indices = np.where(y_pred > threshold)[0]
    best_matches = [filtered_features[i] for i in best_match_indices]

    match = pd.DataFrame(best_matches).sort_values("score", ascending=True).drop_duplicates(["gbif_name"], keep="first")

    df2 = target_dataset.merge(match, left_on='ncbi_target_string', right_on='ncbi_name', how='inner')
    df3 = query_dataset.merge(df2, left_on='gbif_taxonomy', right_on='gbif_name', how='inner')[['taxonID', 'parentNameUsageID', 'canonicalName', 'ncbi_id', 'ncbi_canonicalName', 'prediction', 'score', 'taxonomicStatus', 'gbif_taxonomy', 'ncbi_target_string', 'ncbi_lineage_names', 'ncbi_lineage_ids']]

    initial = set(query)
    matched = set(df3.gbif_taxonomy)
    discarded = list(initial.difference(matched))

    df_matched = df3.copy()
    ncbi_matching = list(set(df_matched.ncbi_id))
    ncbi_missing = target_dataset[~target_dataset.ncbi_id.isin(ncbi_matching)]
    ncbi_missing_2 = ncbi_missing[['ncbi_id', 'ncbi_canonicalName', 'ncbi_target_string', 'ncbi_lineage_names', 'ncbi_lineage_ids']]
    ncbi_missing_3 = target_dataset[target_dataset['ncbi_canonicalName'].str.contains(r'\d')]
    new_df_matched = pd.concat([df_matched, ncbi_missing_2, ncbi_missing_3], ignore_index=True)
    new_df_matched = new_df_matched.fillna(-1)

    
    df_unmatched = query_dataset[query_dataset["gbif_taxonomy"].isin(discarded)]
    return (new_df_matched, df_unmatched)


def filter_synonyms(df_matched, df_unmatched, ncbi_synonyms, gbif_synonyms, gbif_dataset, ncbi_dataset):
    
    """
    Filters the matched dataset to identify and separate synonyms.

    Args:
    [Your existing parameters]

    Returns:
    tuple: DataFrames of filtered synonyms and unmatched entries.
    """ 

    # Filter rows where canonicalName is identical to ncbi_canonicalName
    identical = df_matched.query("canonicalName == ncbi_canonicalName")
    
    # Filter rows where canonicalName is not identical to ncbi_canonicalName
    not_identical = df_matched.query("(canonicalName != ncbi_canonicalName) and taxonID != -1")
    
    # Filter rows where canonicalName is not identical to ncbi_canonicalName
    only_ncbi = df_matched.query("taxonID == -1")
    
    # Calculate Levenshtein distance for non-identical pairs
    lev_dist = not_identical.apply(lambda row: Levenshtein.distance(row['canonicalName'], row['ncbi_canonicalName']), axis=1)
    
    # Create a copy of the filtered DataFrame for non-identical pairs
    similar_pairs = not_identical.copy()
    
    # Add the Levenshtein distance as a new column
    similar_pairs["levenshtein_distance"] = lev_dist
    
    # Filter pairs with a Levenshtein distance less than n
    #similar_pairs_ = similar_pairs[similar_pairs["levenshtein_distance"] < 10]
    
    matching_synonims = []
    excluded_data = []
    
    for index, row in similar_pairs.iterrows():
        taxonID = row['taxonID']
        ncbi_id = row['ncbi_id']
        gbif_canonicalName = row['canonicalName']
        ncbi_canonicalName = row['ncbi_canonicalName']
        
        # Verify if the taxonID is in ncbi_id synonyms
        if gbif_canonicalName in ncbi_synonyms.get(ncbi_id, []):
            matching_synonims.append(row)
        
        # Verify if the ncbi_id is in taxonID synonyms
        elif ncbi_canonicalName in gbif_synonyms[0].get(taxonID, []):
            matching_synonims.append(row)
        
        # Verify if there is a match between gbif and ncbi synonyms
        for gbif_synonym in gbif_synonyms[0].get(taxonID, []):
            for ncbi_synonym in ncbi_synonyms.get(ncbi_id, []):
                if gbif_synonym == ncbi_synonym:
                    matching_synonims.append(row)
        
        #If there is no match, add the row to the excluded data
        if not any(row.equals(existing_row) for existing_row in matching_synonims):
            excluded_data.append(row)
    
    
    # Create separate DataFrame for included and excluded data
    df_matching_synonims = pd.DataFrame(matching_synonims).drop_duplicates()
    df_excluded_ = pd.DataFrame(excluded_data)
    gbif_excluded = gbif_dataset[0][gbif_dataset[0].taxonID.isin(df_excluded_.taxonID)]
    ncbi_excluded = ncbi_dataset[0][ncbi_dataset[0].ncbi_id.isin(df_excluded_.ncbi_id)]
    
    # Concatenate similar pairs with identical samples
    matched_df = pd.concat([identical, df_matching_synonims, only_ncbi, ncbi_excluded])
    matched_df["levenshtein_distance"].fillna(0, inplace=True)
    matched_df = matched_df.sort_values('levenshtein_distance').groupby('ncbi_id').first().reset_index()
    
    # Extract the "gbif_taxonomy" strings from non-similar pairs
    unmatched_df = pd.concat([df_unmatched, gbif_excluded])
    return matched_df, unmatched_df