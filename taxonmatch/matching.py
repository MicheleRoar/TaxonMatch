import re
import os
import random
import pickle
import Levenshtein
import numpy as np
import pandas as pd
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from .model_training import tuple_engineer_features
from .loader import load_gbif_dictionary, load_ncbi_dictionary

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




def extract_after_clade(clade, lineage_names):
    """
    Extracts the portion of the lineage string following the selected clade.

    Args:
    clade (str): The clade to search for.
    lineage_names (str): A semicolon-separated string of lineage names.

    Returns:
    str: A modified lineage string starting from the selected clade.
    """
    # Find the first occurrence of the clade and split the lineage from that point
    parts = lineage_names.split(';')
    try:
        # Start from the index of the clade found in parts
        start_index = parts.index(clade) + 1
        # Create a sub-list starting from the clade and convert it back to a string
        return ';'.join(parts[start_index:]).lower()
    except ValueError:
        # Return an empty string if the clade is not found
        return ""


def select_taxonomic_clade(clade, gbif_dataset, ncbi_dataset):
    """
    Selects and processes entries from the GBIF and NCBI datasets that belong to a specified taxonomic clade.
    Raises an error if the clade is not found in at least one of the datasets.

    Args:
        clade (str): The name of the taxonomic clade to filter for.
        gbif_dataset (DataFrame): The GBIF dataset.
        ncbi_dataset (DataFrame): The NCBI dataset.

    Returns:
        tuple: Processed GBIF and NCBI datasets containing entries of the specified clade.
    """

    # Filter and extract rows of interest from the gbif_dataset DataFrame
    gbif_clade = gbif_dataset[0][gbif_dataset[0]["gbif_taxonomy"].str.contains(";" + clade.lower() + ";")]

    # Filter and apply the extraction function to the DataFrame ncbi_clade
    ncbi_clade = ncbi_dataset[0][ncbi_dataset[0]["ncbi_target_string"].str.contains(";" + clade.lower() + ";")]
    
    if gbif_clade.empty and ncbi_clade.empty:
        raise ValueError(f"The specified clade '{clade}' is not found in either the GBIF or NCBI datasets.")
    elif gbif_clade.empty:
        raise ValueError(f"The specified clade '{clade}' is not found in the GBIF dataset.")
    elif ncbi_clade.empty:
        raise ValueError(f"The specified clade '{clade}' is not found in the NCBI dataset.")

    ncbi_clade_ = ncbi_clade.copy()
    ncbi_clade_.loc[:, 'ncbi_lineage_names'] = clade.lower() + ";" + ncbi_clade_['ncbi_lineage_names'].apply(lambda x: extract_after_clade(clade, x))

    # Update ncbi_lineage_ids to match the number of elements in the new ncbi_lineage_names
    ncbi_clade_['ncbi_lineage_ids'] = ncbi_clade_.apply(lambda row: ';'.join(row['ncbi_lineage_ids'].split(';')[-len(row['ncbi_lineage_names'].split(';')):]), axis=1)

    return gbif_clade, ncbi_clade_



def extract_values_from_dict_list(dict_list):
    feature_matrix = []

    for comparison in dict_list:
        values_from_third = list(comparison.values())[3:] 
        feature_matrix.append(values_from_third)
    
    return feature_matrix


def add_predictions_to_features(features, y_pred):
    for i, prediction in enumerate(y_pred):
        features[i]['prediction'] = prediction


def find_matching(query_dataset, target_dataset, model, relevant_features, threshold):

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


    relevant_features=['levenshtein_distance', 'damerau_levenshtein_distance', 'ratio', 'q_ratio', 'token_sort_ratio', 'w_ratio', 'token_set_ratio', 'jaro_winkler_similarity', 'partial_ratio', 'hamming_distance', 'jaro_similarity']
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


def match_dataset(query_dataset, target_dataset, model, tree_generation = False):
    
    """
    Filters the matched dataset to identify and separate synonyms.
    
    Args:
    [Your existing parameters]
    
    Returns:
    tuple: DataFrames of filtered synonyms and unmatched entries.
    """ 

    relevant_features=['levenshtein_distance', 'damerau_levenshtein_distance', 'ratio', 'q_ratio', 'token_sort_ratio', 'w_ratio', 'token_set_ratio', 'jaro_winkler_similarity', 'partial_ratio', 'hamming_distance', 'jaro_similarity']

    # Carica i dizionari GBIF
    gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids = load_gbif_dictionary()

    # Carica i dizionari NCBI
    ncbi_synonyms_names, ncbi_synonyms_ids = load_ncbi_dictionary()
    
    df_matched, df_unmatched = find_matching(query_dataset, target_dataset, model, relevant_features, 0.90)
    
    # Filter rows where canonicalName is identical to ncbi_canonicalName
    identical = df_matched.query("canonicalName == ncbi_canonicalName")
    
    # Filter rows where canonicalName is not identical to ncbi_canonicalName
    not_identical = df_matched.query("(canonicalName != ncbi_canonicalName) and taxonID != -1")
    
    # Filter rows where canonicalName is not identical to ncbi_canonicalName
    only_ncbi = df_matched.query("taxonID == -1")
    
    matching_synonims = []
    excluded_data = []

    # Pre-elaborazione: converti i nomi in minuscolo una sola volta
    not_identical_ = not_identical.copy()
    not_identical_[['canonicalName', 'ncbi_canonicalName']] = not_identical_[['canonicalName', 'ncbi_canonicalName']].apply(lambda x: x.str.lower())
    gbif_synonyms_lower = {k.lower(): {v.lower() for v in vs} for k, vs in gbif_synonyms_names.items()}
    ncbi_synonyms_lower = {k.lower(): {v.lower() for v in vs} for k, vs in ncbi_synonyms_names.items()}
    
    for index, row in not_identical_.iterrows():
        gbif_canonicalName = row['canonicalName']
        ncbi_canonicalName = row['ncbi_canonicalName']
    
        # Utilizza insiemi per il confronto dei sinonimi
        gbif_synonyms_set = gbif_synonyms_lower.get(gbif_canonicalName, set())
        ncbi_synonyms_set = ncbi_synonyms_lower.get(ncbi_canonicalName, set())
    
        if gbif_canonicalName in ncbi_synonyms_set or ncbi_canonicalName in gbif_synonyms_set or gbif_synonyms_set & ncbi_synonyms_set:
            matching_synonims.append(row)
        else:
            excluded_data.append(row)
    
    # Converti le liste in DataFrame solo dopo il ciclo
    matching_synonims_df = pd.DataFrame(matching_synonims)
    excluded_data_df = pd.DataFrame(excluded_data)

    doubtful = excluded_data_df.copy()
        
    if not doubtful.empty:
        # Calculate Levenshtein distance for non-identical pairs
        lev_dist = doubtful.apply(lambda row: Levenshtein.distance(row['canonicalName'], row['ncbi_canonicalName']), axis=1)
        
        # Create a copy of the filtered DataFrame for non-identical pairs
        similar_pairs = doubtful.copy()
        
        # Add the Levenshtein distance as a new column
        similar_pairs["levenshtein_distance"] = lev_dist
    
        possible_typos_df = pd.DataFrame(similar_pairs).query("levenshtein_distance <= 3").sort_values('score')

        gbif_excluded = query_dataset[query_dataset.taxonID.isin(excluded_data_df.taxonID)]
        ncbi_excluded = target_dataset[target_dataset.ncbi_id.isin(excluded_data_df.ncbi_id)]
    else: 
        possible_typos_df = "No possible typos detected"

    
    # Create separate DataFrame for included and excluded data
    df_matching_synonims = pd.DataFrame(matching_synonims).drop_duplicates()
    
    
    if tree_generation and not doubtful.empty:
        # Concatenate similar pairs with identical samples
        matched_df = pd.concat([identical, df_matching_synonims, only_ncbi, ncbi_excluded])
    else:
        matched_df = pd.concat([identical, df_matching_synonims])
    
    if not doubtful.empty:
        # Extract the "gbif_taxonomy" strings from non-similar pairs
        unmatched_df = pd.concat([df_unmatched, gbif_excluded])
    else:
        unmatched_df = df_unmatched

    matched_df = matched_df.replace([-1, '-1', 'nan', 'NaN', 'none', 'None'], np.nan)
    unmatched_df = unmatched_df.replace([-1, '-1', 'nan', 'NaN', 'none', 'None'], np.nan)

    return matched_df, unmatched_df, possible_typos_df


def find_similar(target_dataset, query_dataset, n_neighbors=3, exclude_perfect_match=True):
    """
    Finds text matches between target and query datasets using TF-IDF vectorization and nearest neighbors.

    Args:
    target_dataset (DataFrame): Dataset containing the target entries with 'ncbi_canonicalName' column.
    query_dataset (DataFrame): Dataset containing the query entries with 'gbif_taxonomy' column.
    n_neighbors (int): Number of nearest neighbors to find. Defaults to 3.
    exclude_perfect_match (bool): Whether to exclude perfect matches (distance = 0). Defaults to True.

    Returns:
    DataFrame: A DataFrame containing the queries, their matched targets, and the distances.
    """

    # Filter entries that contain digits in canonical names
    target_dataset_filtered = target_dataset[~target_dataset['ncbi_canonicalName'].str.contains(r'\d')]
    target = list(set(target_dataset_filtered.ncbi_target_string))
    query = list(set(query_dataset.gbif_taxonomy))

    # Set up the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer=ngrams, lowercase=True)
    tfidf = vectorizer.fit_transform(target)

    # Configure the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1, metric='cosine')
    nbrs.fit(tfidf)

    # Compute the nearest neighbors
    distances, indices = nbrs.kneighbors(vectorizer.transform(query))
    distances = np.round(distances, 2)

    # Prepare lists for the final DataFrame
    filtered_queries = []
    filtered_targets = []
    filtered_distances = []

    for i, (dist_list, idx_list) in enumerate(zip(distances, indices)):
        count = 0
        for dist, idx in zip(dist_list, idx_list):
            if count < n_neighbors and (not exclude_perfect_match or (exclude_perfect_match and dist != 0)):
                filtered_queries.append(query[i])
                filtered_targets.append(target[idx])
                filtered_distances.append(dist)
                count += 1
            if count == n_neighbors:
                break

    # Organize and format the results
    matches_df = pd.DataFrame({
        'Query': filtered_queries,
        'Matched Target': filtered_targets,
        'Distance': filtered_distances
    })

    return matches_df










