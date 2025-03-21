import re
import os
import random
import pickle
import Levenshtein
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from .model_training import compute_similarity_metrics
from .loader import load_gbif_dictionary, load_ncbi_dictionary
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance

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
    parts = lineage_names.lower().split(';')
    try:
        # Start from the index of the clade found in parts
        start_index = parts.index(clade.lower()) + 1
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

    clade_with_boundaries = r'\b' + clade.lower() + r'\b'

    # Filter and extract rows of interest from the gbif_dataset DataFrame
    gbif_clade = gbif_dataset[0][gbif_dataset[0]["gbif_taxonomy"].str.lower().str.contains(clade_with_boundaries.lower() + ";")]

    # Filter and apply the extraction function to the DataFrame ncbi_clade
    ncbi_clade = ncbi_dataset[0][ncbi_dataset[0]["ncbi_lineage_names"].str.lower().str.contains(clade_with_boundaries.lower() + ";")]

    
    if gbif_clade.empty and ncbi_clade.empty:
        raise ValueError(f"The specified clade '{clade}' is not found in either the GBIF or NCBI datasets.")
    elif gbif_clade.empty:
        raise ValueError(f"The specified clade '{clade}' is not found in the GBIF dataset.")
    elif ncbi_clade.empty:
        raise ValueError(f"The specified clade '{clade}' is not found in the NCBI dataset.")

    ncbi_clade_ = ncbi_clade.copy()
    ncbi_clade_.loc[:, 'ncbi_lineage_names'] = clade.lower() + ";" + ncbi_clade_['ncbi_lineage_names'].apply(lambda x: extract_after_clade(clade, x))
    ncbi_clade_.loc[:, 'ncbi_target_string'] = clade.lower() + ";" + ncbi_clade_['ncbi_target_string'].apply(lambda x: extract_after_clade(clade, x))
    
    
    # Update ncbi_lineage_ids to match the number of elements in the new ncbi_lineage_names
    ncbi_clade_['ncbi_lineage_ids'] = ncbi_clade_.apply(lambda row: ';'.join(row['ncbi_lineage_ids'].split(';')[-len(row['ncbi_lineage_names'].split(';')):]), axis=1)
    ncbi_clade_['ncbi_lineage_ranks'] = ncbi_clade_.apply(lambda row: ';'.join(row['ncbi_lineage_ranks'].split(';')[-len(row['ncbi_lineage_names'].split(';')):]), axis=1)

    
    
    gbif_clade_ = gbif_clade.copy()
    gbif_clade_.loc[:, 'gbif_taxonomy'] = clade.lower() + ";" + gbif_clade_['gbif_taxonomy'].apply(lambda x: extract_after_clade(clade, x))
    
    gbif_clade_['gbif_taxonomy_ids'] = gbif_clade_.apply(lambda row: ';'.join(row['gbif_taxonomy_ids'].split(';')[-len(row['gbif_taxonomy'].split(';')):]), axis=1)

    
    return gbif_clade_, ncbi_clade_


def extract_values_from_dict_list(dict_list):
    feature_matrix = []

    for comparison in dict_list:
        values_from_third = list(comparison.values())[3:] 
        feature_matrix.append(values_from_third)
    
    return feature_matrix


def add_predictions_to_features(features, y_pred):
    for i, prediction in enumerate(y_pred):
        features[i]['prediction'] = prediction


def find_neighbors_with_fallback(query, tfidf, vectorizer, max_neighbors=3):
    n_neighbors = max_neighbors
    while n_neighbors > 0:
        try:
            # Try with the current number of neighbors
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1, metric='cosine').fit(tfidf)
            distances, indices = nbrs.kneighbors(vectorizer.transform(query))
            distances = np.round(distances, 2)
            return distances, indices
        except ValueError as e:
            # If it fails, reduce the number of neighbors and try again
            #print(f"Error with n_neighbors={n_neighbors}: {e}")
            n_neighbors -= 1

    # If it's not possible to find neighbors, raise an exception
    raise ValueError("Unable to find neighbors with the provided data.")


def find_matching(query_dataset, target_dataset, model, relevant_features, threshold):
    """
    Matches datasets using a model and calculates the probability of correct matches.

    Args:
    query_dataset (DataFrame): The dataset to query (GBIF).
    target_dataset (DataFrame): The dataset to match against (NCBI).
    model: The machine learning model used for matching.
    relevant_features: Features relevant to the matching process.
    threshold (float): The threshold for determining a match.

    Returns:
    tuple: DataFrames of matched and unmatched entries.
    """

    # Remove taxa with numbers from target dataset (these will be reintroduced later)
    target_dataset_filtered = target_dataset[~target_dataset['ncbi_canonicalName'].str.contains(r'\d')]

    # Extract unique taxonomic names
    target = list(set(target_dataset_filtered.ncbi_target_string))
    query = list(set(query_dataset.gbif_taxonomy))

    # Set up the TF-IDF vectorizer for textual similarity
    vectorizer = TfidfVectorizer(analyzer=ngrams, lowercase=True)
    tfidf = vectorizer.fit_transform(target)

    # Find neighbors with TF-IDF scores
    try:
        distances, indices = find_neighbors_with_fallback(query, tfidf, vectorizer, max_neighbors=3)
    except ValueError as e:
        print(f"Error during neighbor search: {e}")
        return None  # Return None if an error occurs

    # Expand results to align query taxa with their closest matches
    expanded_distances = distances.flatten()
    expanded_query = np.repeat(query, distances.shape[1])
    expanded_target = np.array(target)[indices.flatten()]

    matches = np.column_stack((expanded_distances, expanded_query, expanded_target))

    # Convert matches to DataFrame
    match_df = pd.DataFrame(matches, columns=['distance', 'query_name', 'target_name'])

    # Compute similarity features directly
    match_df = match_df.merge(query_dataset[["gbif_taxonomy", "taxonRank"]], 
                              left_on="query_name", right_on="gbif_taxonomy", how="left")\
                       .merge(target_dataset[["ncbi_target_string", "ncbi_rank"]], 
                              left_on="target_name", right_on="ncbi_target_string", how="left")

    match_df = match_df[["query_name", "target_name", "distance", "taxonRank", "ncbi_rank"]]

    similarity_features = compute_similarity_metrics(match_df)

    # Predict match probabilities using the model
    X_features = similarity_features[relevant_features]
    y_pred = model.predict_proba(X_features)[:, 1]
    match_df['prediction'] = y_pred

    # Select best matches exceeding the threshold
    #match_df = match_df[y_pred > threshold]
    match_df = match_df.sort_values("prediction", ascending=False).drop_duplicates(["query_name"], keep="first")

    df2 = target_dataset.merge(match_df, left_on='ncbi_target_string', right_on='target_name', how='inner', suffixes=("_target", "_match"))
    # Select only the columns of match_df that have the _match suffix and remove the original ones.
    df2 = df2.drop(columns=["ncbi_rank_target"]).rename(columns={"ncbi_rank_match": "ncbi_rank"})

    df3 = query_dataset.merge(df2, left_on='gbif_taxonomy', right_on='query_name', how='inner', suffixes=("_target", "_match"))
    df3 = df3.drop(columns=["taxonRank_target"]).rename(columns={"taxonRank_match": "taxonRank"})

    # Merge with query dataset and retain necessary columns
    df3 = df3[
        ['taxonID', 'parentNameUsageID', 'canonicalName', 'ncbi_id', 'ncbi_canonicalName', 
         'taxonRank', 'ncbi_rank', 'prediction', 'distance', 'taxonomicStatus', 
         'gbif_taxonomy', 'ncbi_target_string', 'ncbi_lineage_names', 'ncbi_lineage_ids']
    ]

    # Identify unmatched taxa
    initial = set(query)
    matched = set(df3.gbif_taxonomy)
    discarded = list(initial.difference(matched))

    # Retrieve unmatched target taxa
    ncbi_matching = list(set(df3.ncbi_id))
    ncbi_missing = target_dataset[~target_dataset.ncbi_id.isin(ncbi_matching)]
    ncbi_missing_df = ncbi_missing[['ncbi_id', 'ncbi_canonicalName', 'ncbi_target_string', 'ncbi_lineage_names', 'ncbi_lineage_ids']]
    
    # Restore taxa with numbers (previously removed)
    ncbi_missing_restored = target_dataset[target_dataset['ncbi_canonicalName'].str.contains(r'\d')]

    # Combine matched taxa with unmatched target taxa
    new_df_matched = pd.concat([df3, ncbi_missing_df, ncbi_missing_restored], ignore_index=True).fillna(-1)

    # Extract unmatched query taxa
    df_unmatched = query_dataset[query_dataset["gbif_taxonomy"].isin(discarded)]

    return new_df_matched, df_unmatched




def match_dataset(query_dataset, target_dataset, model, tree_generation = False): 
    """
    Filters the matched dataset to identify and separate synonyms.
    
    Args:
    [Your existing parameters]
    
    Returns:
    tuple: DataFrames of filtered synonyms and unmatched entries.
    """ 
 
    #notes:
    #generalize input column labels
    #canonicalName taxonID -> query_canonicalName query_taxonID
    #ncbi_canonicalName, ncbi_id -> target_canonicalName, target_id, 

    
    relevant_features=['rank_similarity', 'levenshtein_distance', 'damerau_levenshtein_distance', 'ratio', 'q_ratio', 'token_sort_ratio', 'w_ratio', 'token_set_ratio', 'jaro_winkler_similarity', 'partial_ratio', 'hamming_distance', 'jaro_similarity']
    
    # Load GBIF dictionary
    gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids = load_gbif_dictionary()
    
    # Load NCBI dictionary
    ncbi_synonyms_names, ncbi_synonyms_ids = load_ncbi_dictionary()
    
    df_matched, df_unmatched = find_matching(query_dataset, target_dataset, model, relevant_features, 0.20)
    
    # Filter rows where canonicalName is identical to ncbi_canonicalName
    identical = df_matched.query("canonicalName == ncbi_canonicalName")
    
    # Filter rows where canonicalName is not identical to ncbi_canonicalName
    not_identical = df_matched.query("(canonicalName != ncbi_canonicalName) and taxonID != -1")

    # Filter rows where canonicalName is not identical to ncbi_canonicalName
    only_ncbi = df_matched.query("(taxonID == -1) and ncbi_lineage_ranks != -1")
    
    matching_synonims = []
    excluded_data = []
    
    # Preprocessing: Convert names to lowercase only once
    not_identical_ = not_identical.copy()
    not_identical_[['canonicalName', 'ncbi_canonicalName']] = not_identical_[['canonicalName', 'ncbi_canonicalName']].apply(lambda x: x.str.lower())
    gbif_synonyms_lower = {k.lower(): {v.lower() for v in vs} for k, vs in gbif_synonyms_names.items()}
    ncbi_synonyms_lower = {k.lower(): {v.lower() for v in vs} for k, vs in ncbi_synonyms_names.items()}
    
    for index, row in not_identical_.iterrows():
        gbif_canonicalName = row['canonicalName']
        ncbi_canonicalName = row['ncbi_canonicalName']
    
        # Use sets for synonym comparison
        gbif_synonyms_set = gbif_synonyms_lower.get(gbif_canonicalName, set())
        ncbi_synonyms_set = ncbi_synonyms_lower.get(ncbi_canonicalName, set())
    
        if gbif_canonicalName in ncbi_synonyms_set or ncbi_canonicalName in gbif_synonyms_set or gbif_synonyms_set & ncbi_synonyms_set:
            matching_synonims.append(row)
        else:
            excluded_data.append(row)
    
    # Convert lists to DataFrame only after loop
    excluded_data_df = pd.DataFrame(excluded_data)
    doubtful = excluded_data_df.copy()
        
    if not doubtful.empty:
        doubtful = doubtful.dropna(subset=['canonicalName', 'ncbi_canonicalName'])
        # Calculate Levenshtein distance for non-identical pairs
        lev_dist = doubtful.apply(lambda row: Levenshtein.distance(row['canonicalName'], row['ncbi_canonicalName']), axis=1)
    
        # Create a copy of the filtered DataFrame for non-identical pairs
        similar_pairs = doubtful.copy()
        
        # Add the Levenshtein distance as a new column
        similar_pairs["levenshtein_distance"] = lev_dist
    
        possible_typos_df = pd.DataFrame(similar_pairs).query("levenshtein_distance <= 3").sort_values('distance')
        
        gbif_excluded = query_dataset[query_dataset.taxonID.isin(doubtful.taxonID)]
        ncbi_excluded = target_dataset[target_dataset.ncbi_id.isin(doubtful.ncbi_id)]
    else: 
        possible_typos_df = "No possible typos detected"
    


    # Create separate DataFrame for included and excluded data
    try:
        df_matching_synonims = pd.DataFrame(matching_synonims).drop_duplicates()
        df_matching_synonims.loc[:, 'ncbi_id'] = df_matching_synonims['ncbi_id'].astype(int)
    except KeyError as e:
        df_matching_synonims = pd.DataFrame() 
    
    # Assuming you have your sets defined
    iden = set(identical.ncbi_id)
    
    if not doubtful.empty:
        # Filter out the excluded IDs from other DataFrames
        ncbi_excluded_filtered = ncbi_excluded[~ncbi_excluded.ncbi_id.isin(iden)]

    
    if tree_generation and not doubtful.empty:
        # Concatenate similar pairs with identical samples
        matched_df = pd.concat([identical , df_matching_synonims, only_ncbi, ncbi_excluded_filtered])
    else:
        matched_df = pd.concat([identical , df_matching_synonims])
        
    matched_df = matched_df.infer_objects(copy=False).fillna(-1)
    matched_df['taxonID'] = matched_df['taxonID'].astype(int)
    
    if not doubtful.empty:
        # Extract the "gbif_taxonomy" strings from non-similar pairs
        unmatched_df = pd.concat([df_unmatched, gbif_excluded])
    else:
        unmatched_df = df_unmatched
    
    unmatched_df = unmatched_df.infer_objects(copy=False).fillna(-1)
    
    matched_df = matched_df.replace([-1, '-1'], None)
    unmatched_df = unmatched_df.replace([-1, '-1'], None)
    return matched_df, unmatched_df, possible_typos_df



def find_top_n_similar(input_string, target_dataset, n_neighbors=3):
    """
    Finds the top N similar strings in the target dataset for a given input string using TF-IDF vectorization and nearest neighbors.

    Args:
    input_string (str): The input string to match against the target dataset.
    target_dataset (DataFrame): Dataset containing the target entries with 'ncbi_canonicalName' column.
    n_neighbors (int): Number of nearest neighbors to find. Defaults to 3.

    Returns:
    DataFrame: A DataFrame containing the matched targets and the distances.
    """

    # Filter entries that contain digits in canonical names
    target_dataset_filtered = target_dataset[~target_dataset['ncbi_canonicalName'].str.contains(r'\d')]
    target = list(set(target_dataset_filtered.ncbi_canonicalName))

    # Set up the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer=ngrams, lowercase=True)
    tfidf = vectorizer.fit_transform(target)

    # Configure the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1, metric='cosine')
    nbrs.fit(tfidf)

    # Compute the nearest neighbors for the input string
    distances, indices = nbrs.kneighbors(vectorizer.transform([input_string]))
    distances = np.round(distances[0], 2)
    indices = indices[0]

    # Prepare lists for the final DataFrame
    matched_targets = [target[idx] for idx in indices]
    matched_ids = [target_dataset_filtered[target_dataset_filtered['ncbi_canonicalName'] == target[idx]]['ncbi_id'].values[0] for idx in indices]
    matched_distances = distances

    # Organize and format the results
    matches_df = pd.DataFrame({
        'Query': [input_string] * len(matched_targets),
        'ncbi_id': matched_ids,
        'Matched Target': matched_targets,
        'Distance': matched_distances
    })

    return matches_df



def find_closest_sample(target_dataset, query_dataset, n_neighbors=1, similarity_threshold=0.70, max_levenshtein_distance=2):
    """
    Finds the nearest neighbor in the target dataset for each string in the query dataset using TF-IDF vectorization and nearest neighbors,
    with an option to filter based on a similarity threshold and the Levenshtein distance of the first word after splitting by the last semicolon.

    Args:
    target_dataset (DataFrame): Dataset containing the target entries with 'ncbi_canonicalName' and 'ncbi_id' columns.
    query_dataset (DataFrame): Dataset containing the query entries with 'gbif_taxonomy' column.
    n_neighbors (int): Number of nearest neighbors to find. Defaults to 1.
    similarity_threshold (float): Threshold for similarity. Entries with distances above this threshold are discarded. Defaults to 0.70.
    max_levenshtein_distance (int): Maximum allowed Levenshtein distance for the first word comparison. Defaults to 3.

    Returns:
    DataFrame: A DataFrame containing the queries, their matched targets, their IDs, and the distances.
    """

    # Filter entries that contain digits in canonical names
    target_dataset_filtered = target_dataset[~target_dataset['ncbi_canonicalName'].str.contains(r'\d')]
    target = list(target_dataset_filtered.ncbi_canonicalName)
    target_ids = list(target_dataset_filtered.ncbi_id)
    query = list(query_dataset.gbif_taxonomy)

    # Set up the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer=ngrams, lowercase=True)
    tfidf = vectorizer.fit_transform(target)

    # Configure the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1, metric='cosine')
    nbrs.fit(tfidf)

    # Compute the nearest neighbors for the query strings
    distances, indices = nbrs.kneighbors(vectorizer.transform(query))
    distances = np.round(distances, 2)

    # Prepare lists for the final DataFrame
    filtered_queries = []
    filtered_targets = []
    filtered_ids = []
    filtered_distances = []

    for i, (dist_list, idx_list) in enumerate(zip(distances, indices)):
        if dist_list[0] <= similarity_threshold:
            query_string = query[i]
            target_string = target[idx_list[0]]

            # Extract the last part after the last semicolon in query and target
            query_last_part = query_string.split(';')[-1].split()[0].lower()
            target_last_part = target_string.split(';')[-1].split()[0].lower()

            # Calculate the Levenshtein distance between the first words
            lev_distance = Levenshtein.distance(query_last_part, target_last_part)
            
            if lev_distance <= max_levenshtein_distance:
                filtered_queries.append(query_string)
                filtered_targets.append(target_string)
                filtered_ids.append(target_ids[idx_list[0]])
                filtered_distances.append(dist_list[0])

    # Organize and format the results
    matches_df = pd.DataFrame({
        'Query': filtered_queries,
        'Matched_id': filtered_ids,
        'Matched Target': filtered_targets,
        'Distance': filtered_distances
    })

    return matches_df







# Step 1: Exact matches + best authorship selection
def get_exact_matches(query_dataset, target_dataset, column):
    target_names = target_dataset[["canonicalName", "taxonID", "scientificNameAuthorship", "taxonomicStatus"]].drop_duplicates().fillna("Not Available")
    exact_matches = query_dataset[query_dataset[column].isin(target_names["canonicalName"])].copy()
    exact_matches = exact_matches.merge(target_names, left_on=column, right_on="canonicalName", how="left")
    exact_matches["Distance"] = 0

    grouped = exact_matches.groupby(column, group_keys=False)
    exact_matches_clean = []

    for group_key, group_df in grouped:
        group_df = group_df.drop(columns=[column])
        group_df[column] = group_key
        best_row = select_best_authorship(group_df)
        exact_matches_clean.append(best_row)

    return pd.concat(exact_matches_clean, ignore_index=True)

# Step 1b: Best authorship match
def select_best_authorship(df):
    df = df.copy()
    if "L." in df["scientificNameAuthorship"].values:
        return df[df["scientificNameAuthorship"] == "L."].head(1)
    else:
        if "AuthorshipMatchScore" not in df.columns:
            return df.head(1)
        return df.sort_values(by=["AuthorshipMatchScore", "Distance"], ascending=[False, True]).head(1)

# Step 2: Get top TF-IDF candidates
def get_top_candidates(taxon, target_names, tfidf_matrix, vectorizer, top_n):
    taxon_tfidf = vectorizer.transform([taxon])
    similarities = cosine_similarity(taxon_tfidf, tfidf_matrix).flatten()
    top_candidates_idx = np.argsort(similarities)[-top_n:][::-1]
    return target_names.iloc[top_candidates_idx].copy()

# Step 3: Score candidates including authorship distance
def score_candidates(taxon, candidates, reference_authorship):
    taxon_parts = taxon.split()
    taxon_word_count = len(taxon_parts)
    taxon_first_word = taxon_parts[0]

    candidates["NameDistance"] = candidates["canonicalName"].apply(lambda x: -distance(taxon, x))
    candidates["AuthorshipMatchScore"] = candidates["scientificNameAuthorship"].apply(
        lambda x: -distance(x, reference_authorship)
    )

    candidates = candidates[
        candidates["canonicalName"].apply(lambda x: distance(x.split(" ")[0], taxon_first_word) <= 3)
    ].copy()

    candidates["WordCount"] = candidates["canonicalName"].apply(lambda x: len(x.split(" ")))
    candidates["AdjustedNameDistance"] = candidates.apply(
        lambda row: row["NameDistance"] - abs(row["WordCount"] - taxon_word_count) * 2,
        axis=1
    )

    if taxon_word_count == 3:
        taxon_infraspecific = taxon_parts[2]

        def penalize_infraspecific(row):
            parts = row["canonicalName"].split()
            if len(parts) == 3:
                dist = distance(parts[2], taxon_infraspecific)
                return row["AdjustedNameDistance"] - dist * 2
            return row["AdjustedNameDistance"] - 5

        candidates["AdjustedNameDistance"] = candidates.apply(penalize_infraspecific, axis=1)

    return candidates.sort_values(by=["AdjustedNameDistance"], ascending=False)

# Step 4: Return top match
def find_best_match(taxon, candidates):
    best_match = candidates.head(1)
    return (
        taxon,
        best_match["canonicalName"].iloc[0],
        best_match["taxonID"].iloc[0],
        best_match["NameDistance"].iloc[0],
        best_match["scientificNameAuthorship"].iloc[0],
        best_match["AuthorshipMatchScore"].iloc[0]
    )

# Main pipeline

def find_gbif_similar_taxa(query_dataset, target_dataset, column, top_n=3):
    # Setup
    max_name_penalty = 30  # maximum distance considered for 0% match
    target_names = target_dataset[["canonicalName", "taxonID", "scientificNameAuthorship", "taxonomicStatus"]].drop_duplicates().fillna("Not Available")
    vectorizer = TfidfVectorizer(analyzer=ngrams, lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(target_names["canonicalName"])

    # Step 1: exact matches
    exact_matches = get_exact_matches(query_dataset, target_dataset, column)
    exact_matches = exact_matches.rename(columns={
        "canonicalName": "gbif_canonicalName",
        "taxonID": "gbif_taxonID",
        "scientificNameAuthorship": "gbif_scientificNameAuthorship"
    })
    exact_matches["MatchScore"] = 100.0  # exact match is always 100%

    # Step 2: approximate matches
    taxa_to_match = query_dataset[~query_dataset[column].isin(target_names["canonicalName"])].copy()
    results = []

    reference_col = column.replace("_cleaned", "")
    for _, row in taxa_to_match.iterrows():
        taxon = row[column]
        reference_authorship = row[reference_col] if reference_col in row and pd.notna(row[reference_col]) else ""
        candidates = get_top_candidates(taxon, target_names, tfidf_matrix, vectorizer, top_n)
        scored_candidates = score_candidates(taxon, candidates, reference_authorship)
        if not scored_candidates.empty:
            results.append(find_best_match(taxon, scored_candidates))

    df_approx_matches = pd.DataFrame(results, columns=[
        column,
        "gbif_canonicalName",
        "gbif_taxonID",
        "Distance",
        "gbif_scientificNameAuthorship",
        "AuthorshipMatchScore"
    ])

    df_approx_final = df_approx_matches.merge(query_dataset, on=column, how="left")
    df_approx_final["MatchScore"] = 100 * (1 - abs(df_approx_final["Distance"]) / max_name_penalty)
    df_approx_final["MatchScore"] = df_approx_final["MatchScore"].clip(lower=0, upper=100).astype(int)

    # Align both DataFrames
    common_cols = [
        'Family', 'TAXON', 'Threat Category', 'Criteria', 'Threats',
        'TAXON_cleaned', 'gbif_canonicalName', 'gbif_taxonID',
        'gbif_scientificNameAuthorship', 'MatchScore'
    ]

    for col in common_cols:
        if col not in exact_matches.columns:
            exact_matches[col] = None
        if col not in df_approx_final.columns:
            df_approx_final[col] = None

    exact_matches = exact_matches[common_cols]
    df_approx_final = df_approx_final[common_cols]
    final_df = pd.concat([exact_matches, df_approx_final], ignore_index=True)

    return final_df


def add_gbif_synonyms(df):
    
    df = df.copy()
    
    gbif_synonym_names, gbif_synonyms_ids, gbif_synonyms_tuples = load_gbif_dictionary()

    df.loc[:, 'gbif_synonyms_names'] = df['gbif_taxonID'].map(
        lambda x: '; '.join([name for name, _ in gbif_synonyms_tuples.get(x, [])])
    )

    df.loc[:, 'gbif_synonyms_ids'] = df['gbif_taxonID'].map(
        lambda x: '; '.join([str(syn_id) for _, syn_id in gbif_synonyms_tuples.get(x, [])])
    )
    
    return df
