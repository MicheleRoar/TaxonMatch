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

    # Force ncbi_id and taxonID to integers early
    target_dataset['ncbi_id'] = pd.to_numeric(target_dataset['ncbi_id'], errors='coerce').fillna(-1).astype(int)
    query_dataset['taxonID'] = pd.to_numeric(query_dataset['taxonID'], errors='coerce').fillna(-1).astype(int)

    # Remove taxa with numbers from the target dataset
    target_dataset_filtered = target_dataset[~target_dataset['ncbi_canonicalName'].str.contains(r'\d', na=False)]

    # Unique taxonomic names
    target = list(set(target_dataset_filtered['ncbi_target_string']))
    query = list(set(query_dataset['gbif_taxonomy']))

    # Vectorization for text similarity
    vectorizer = TfidfVectorizer(analyzer=ngrams, lowercase=True)
    tfidf = vectorizer.fit_transform(target)

    # Find nearest neighbors
    try:
        distances, indices = find_neighbors_with_fallback(query, tfidf, vectorizer, max_neighbors=3)
    except ValueError as e:
        print(f"Error during neighbor search: {e}")
        return None, None

    # Expand matches
    expanded_distances = distances.flatten()
    expanded_query = np.repeat(query, distances.shape[1])
    expanded_target = np.array(target)[indices.flatten()]

    matches = pd.DataFrame({
        'distance': expanded_distances,
        'query_name': expanded_query,
        'target_name': expanded_target
    })

    # Merge additional features for prediction
    matches = matches.merge(query_dataset[['gbif_taxonomy', 'taxonRank']], 
                             left_on='query_name', right_on='gbif_taxonomy', how='left')\
                     .merge(target_dataset[['ncbi_target_string', 'ncbi_rank']], 
                             left_on='target_name', right_on='ncbi_target_string', how='left')

    matches = matches[['query_name', 'target_name', 'distance', 'taxonRank', 'ncbi_rank']]

    # Compute similarity features
    similarity_features = compute_similarity_metrics(matches)

    # Predict match probabilities
    X_features = similarity_features[relevant_features]
    y_pred = model.predict_proba(X_features)[:, 1]
    matches['prediction'] = y_pred

    # Keep only best matches
    matches = matches.sort_values('prediction', ascending=False).drop_duplicates('query_name', keep='first')

    # Merge with full data
    matched_target = target_dataset.merge(matches, 
                                          left_on='ncbi_target_string', right_on='target_name', 
                                          how='inner', suffixes=('_target', '_match')) \
                                   .drop(columns=['ncbi_rank_target']) \
                                   .rename(columns={'ncbi_rank_match': 'ncbi_rank'})

    matched_full = query_dataset.merge(matched_target, 
                                       left_on='gbif_taxonomy', right_on='query_name', 
                                       how='inner', suffixes=('_target', '_match')) \
                                .drop(columns=['taxonRank_target']) \
                                .rename(columns={'taxonRank_match': 'taxonRank'})

    # Select final columns
    matched_full = matched_full[[
        'taxonID', 'parentNameUsageID', 'canonicalName', 'ncbi_id', 'ncbi_canonicalName',
        'taxonRank', 'ncbi_rank', 'prediction', 'distance', 'taxonomicStatus',
        'gbif_taxonomy', 'ncbi_target_string', 'ncbi_lineage_names', 'ncbi_lineage_ids'
    ]]

    # Identify unmatched taxa
    initial_query_set = set(query)
    matched_query_set = set(matched_full['gbif_taxonomy'])
    discarded_queries = list(initial_query_set.difference(matched_query_set))

    # Find unmatched target taxa
    matched_ncbi_ids = matched_full['ncbi_id'].fillna(-1).astype(int)
    ncbi_missing = target_dataset[~target_dataset['ncbi_id'].isin(matched_ncbi_ids)]

    # Restore NCBI entries with numbers (previously removed)
    ncbi_missing_with_numbers = target_dataset[target_dataset['ncbi_canonicalName'].str.contains(r'\d', na=False)]

    # Combine all matched entries
    final_matched_df = pd.concat([matched_full, ncbi_missing, ncbi_missing_with_numbers], ignore_index=True).fillna(-1)

    # Extract unmatched query entries
    unmatched_queries_df = query_dataset[query_dataset['gbif_taxonomy'].isin(discarded_queries)]

    return final_matched_df, unmatched_queries_df



def append_missing_ncbi_entries(matched_df, target_dataset):
    matched_ncbi_ids = set(matched_df['ncbi_id'].dropna().astype(int))
    all_ncbi_ids = set(target_dataset['ncbi_id'].dropna().astype(int))
    missing_ncbi_ids = all_ncbi_ids - matched_ncbi_ids

    missing_ncbi_rows = target_dataset[target_dataset['ncbi_id'].astype(int).isin(missing_ncbi_ids)].copy()

    if missing_ncbi_rows.empty:
        return matched_df

    for col in ['taxonID', 'parentNameUsageID', 'canonicalName', 'taxonRank', 'prediction',
                'distance', 'taxonomicStatus', 'gbif_taxonomy']:
        missing_ncbi_rows[col] = None

    missing_ncbi_rows = missing_ncbi_rows[matched_df.columns]
    return pd.concat([matched_df, missing_ncbi_rows], ignore_index=True)



def append_missing_gbif_entries(matched_df, unmatched_df, query_dataset):
    matched_gbif_ids = set(matched_df['taxonID'].dropna().astype(int))
    unmatched_gbif_ids = set(unmatched_df['taxonID'].dropna().astype(int))
    existing_gbif_ids = matched_gbif_ids.union(unmatched_gbif_ids)

    all_gbif_ids = set(query_dataset['taxonID'].dropna().astype(int))
    missing_gbif_ids = all_gbif_ids - existing_gbif_ids

    missing_gbif_rows = query_dataset[query_dataset['taxonID'].astype(int).isin(missing_gbif_ids)].copy()

    if missing_gbif_rows.empty:
        return unmatched_df

    return pd.concat([unmatched_df, missing_gbif_rows], ignore_index=True)



def match_dataset(query_dataset, target_dataset, model, tree_generation=False):
    """
    Match taxonomic entries between a query dataset (GBIF) and a target dataset (NCBI),
    identify exact matches, synonyms, and potential typos. Ensures sample completeness
    using reference datasets (if provided).

    Args:
        query_dataset (DataFrame): GBIF input dataset.
        target_dataset (DataFrame): NCBI target dataset.
        model: Trained ML model for similarity.
        tree_generation (bool): If True, include doubtful matches.

    Returns:
        matched_df, unmatched_df, possible_typos_df
    """

    relevant_features = [
        'rank_similarity', 'levenshtein_distance', 'damerau_levenshtein_distance', 'ratio',
        'q_ratio', 'token_sort_ratio', 'w_ratio', 'token_set_ratio', 'jaro_winkler_similarity',
        'partial_ratio', 'hamming_distance', 'jaro_similarity'
    ]

    gbif_synonyms_names, _, _ = load_gbif_dictionary()
    ncbi_synonyms_names, _ = load_ncbi_dictionary()

    df_matched, df_unmatched = find_matching(query_dataset, target_dataset, model, relevant_features, 0.20)

    df_matched["canonicalName"] = df_matched["canonicalName"].astype(str).str.strip()
    df_matched["ncbi_canonicalName"] = df_matched["ncbi_canonicalName"].astype(str).str.strip()

    identical = df_matched[
        (df_matched["canonicalName"] == df_matched["ncbi_canonicalName"]) &
        (df_matched["canonicalName"] != "-1")
    ]

    not_identical = df_matched[
        (df_matched["canonicalName"] != df_matched["ncbi_canonicalName"]) &
        (df_matched["canonicalName"] != "-1") &
        (df_matched["ncbi_canonicalName"] != "-1")
    ]

    used_idx = set(identical.index) | set(not_identical.index)
    only_ncbi = df_matched.loc[~df_matched.index.isin(used_idx)]

    not_identical_ = not_identical.copy()
    not_identical_[['canonicalName', 'ncbi_canonicalName']] = not_identical_[
        ['canonicalName', 'ncbi_canonicalName']
    ].apply(lambda x: x.str.lower())

    gbif_synonyms_lower = {k.lower(): {v.lower() for v in vs} for k, vs in gbif_synonyms_names.items()}
    ncbi_synonyms_lower = {k.lower(): {v.lower() for v in vs} for k, vs in ncbi_synonyms_names.items()}

    matching_synonyms = []
    excluded_data = []

    for _, row in not_identical_.iterrows():
        gbif_name = row['canonicalName']
        ncbi_name = row['ncbi_canonicalName']
        gbif_set = gbif_synonyms_lower.get(gbif_name, set())
        ncbi_set = ncbi_synonyms_lower.get(ncbi_name, set())

        if gbif_name in ncbi_set or ncbi_name in gbif_set or gbif_set & ncbi_set:
            matching_synonyms.append(row)
        else:
            excluded_data.append(row)

    doubtful = pd.DataFrame(excluded_data)


    if not doubtful.empty:
        doubtful = doubtful.dropna(subset=['canonicalName', 'ncbi_canonicalName'])
        lev_dist = doubtful.apply(
            lambda row: Levenshtein.distance(row['canonicalName'], row['ncbi_canonicalName']),
            axis=1
        )
        similar_pairs = doubtful.copy()
        similar_pairs["levenshtein_distance"] = lev_dist
        possible_typos_df = similar_pairs.query("levenshtein_distance <= 3").sort_values('levenshtein_distance')

        gbif_excluded = query_dataset[query_dataset['taxonID'].isin(doubtful['taxonID'])]
        ncbi_excluded = target_dataset[target_dataset['ncbi_id'].isin(doubtful['ncbi_id'])]
    else:
        possible_typos_df = "No possible typos detected"
        gbif_excluded = pd.DataFrame(columns=query_dataset.columns)
        ncbi_excluded = pd.DataFrame(columns=target_dataset.columns)

    df_matching_synonyms = pd.DataFrame(matching_synonyms).drop_duplicates() if matching_synonyms else pd.DataFrame()

    matched_df = pd.concat([identical, df_matching_synonyms])
    iden = set(identical['ncbi_id'])

    if tree_generation and not doubtful.empty:
        ncbi_excluded_filtered = ncbi_excluded[~ncbi_excluded['ncbi_id'].isin(iden)]
        matched_df = pd.concat([matched_df, only_ncbi, ncbi_excluded_filtered])
    else:
        matched_df = pd.concat([matched_df, only_ncbi])

    matched_df = matched_df.infer_objects(copy=False).fillna(-1)
    matched_df['taxonID'] = pd.to_numeric(matched_df['taxonID'], errors='coerce').astype('Int64')

    unmatched_df = pd.concat([df_unmatched, gbif_excluded]) if not doubtful.empty else df_unmatched
    unmatched_df = unmatched_df.infer_objects(copy=False).fillna(-1)

    matched_df = matched_df.replace([-1, '-1'], None)
    unmatched_df = unmatched_df.replace([-1, '-1'], None)

    matched_df['ncbi_id'] = pd.to_numeric(matched_df['ncbi_id'], errors='coerce').astype('Int64')
    matched_df['distance'] = pd.to_numeric(matched_df['distance'], errors='coerce')

    # Deduplicate
    duplicated_ids = matched_df['ncbi_id'][matched_df['ncbi_id'].duplicated(keep=False)]
    dups = matched_df[matched_df['ncbi_id'].isin(duplicated_ids)]
    dups_sorted = dups.sort_values(by=['ncbi_id', 'distance'], key=lambda col: col.isna().astype(int))
    dups_best = dups_sorted.drop_duplicates(subset='ncbi_id', keep='first')
    matched_df = matched_df[~matched_df.index.isin(dups.index)]
    matched_df = pd.concat([matched_df, dups_best], ignore_index=True)

    # Add missing rows if reference datasets are provided
    if target_dataset is not None:
        matched_df = append_missing_ncbi_entries(matched_df, target_dataset)

    if query_dataset is not None:
        unmatched_df = append_missing_gbif_entries(matched_df, unmatched_df, query_dataset)

    return matched_df, unmatched_df, possible_typos_df



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

    return add_gbif_synonyms(final_df)


def select_closest_common_clade(species_name, gbif_data, ncbi_data):
    import re

    if isinstance(gbif_data, (list, tuple)):
        gbif_data = gbif_data[0]
    if isinstance(ncbi_data, (list, tuple)):
        ncbi_data = ncbi_data[0]

    def extract_after_clade(clade, lineage_string):
        parts = lineage_string.lower().split(';')
        if clade not in parts:
            return ''
        idx = parts.index(clade)
        return ';'.join(parts[idx + 1:])

    def safe_trim_ids(ids_str, names_str):
        ids = ids_str.split(';')
        names = names_str.split(';')
        if len(ids) < len(names):
            print(f"[Mismatch] {len(names)} names vs {len(ids)} ids")
            return ';'.join(ids)  # fallback: restituisci tutto
        return ';'.join(ids[-len(names):])

    # Recupera la riga della specie se presente
    gbif_row = gbif_data[gbif_data["canonicalName"].str.lower() == species_name.lower()]
    ncbi_row = ncbi_data[ncbi_data["ncbi_canonicalName"].str.lower() == species_name.lower()]

    if gbif_row.empty and ncbi_row.empty:
        raise ValueError(f"Species '{species_name}' not found in either dataset.")

    # Estrai il lineage GBIF o NCBI per scorrere a ritroso
    gbif_lineage = gbif_row.iloc[0]["gbif_taxonomy"].lower().split(";")[:-1] if not gbif_row.empty else []
    ncbi_lineage = ncbi_row.iloc[0]["ncbi_lineage_names"].lower().split(";")[:-1] if not ncbi_row.empty else []

    lineage_to_search = gbif_lineage if gbif_lineage else ncbi_lineage

    for clade in reversed(lineage_to_search):
        pattern = r'\b' + re.escape(clade) + r'\b;'
        gbif_clade = gbif_data[gbif_data["gbif_taxonomy"].str.lower().str.contains(pattern)]
        ncbi_clade = ncbi_data[ncbi_data["ncbi_lineage_names"].str.lower().str.contains(pattern)]

        gbif_species = gbif_clade["canonicalName"].str.lower().unique().tolist()
        ncbi_species = ncbi_clade["ncbi_canonicalName"].str.lower().unique().tolist()

        if len(gbif_species) > 1 or len(ncbi_species) > 1:
            print(f"Last common node: {clade}")

            # Ricostruisci GBIF
            gbif_clade_ = gbif_clade.copy()
            gbif_clade_["gbif_taxonomy"] = clade + ";" + gbif_clade_["gbif_taxonomy"].apply(lambda x: extract_after_clade(clade, x))
            gbif_clade_["gbif_taxonomy_ids"] = gbif_clade_.apply(
                lambda row: ';'.join(row["gbif_taxonomy_ids"].split(";")[-len(row["gbif_taxonomy"].split(";")):]), axis=1
            )

            # Ricostruisci NCBI
            ncbi_clade_ = ncbi_clade.copy()
            ncbi_clade_["ncbi_lineage_names"] = clade + ";" + ncbi_clade_["ncbi_lineage_names"].apply(lambda x: extract_after_clade(clade, x))
            ncbi_clade_["ncbi_target_string"] = clade + ";" + ncbi_clade_["ncbi_target_string"].apply(lambda x: extract_after_clade(clade, x))

            # Usa safe_trim_ids per trimming robusto
            ncbi_clade_["ncbi_lineage_ids"] = ncbi_clade_.apply(
                lambda row: safe_trim_ids(row["ncbi_lineage_ids"], row["ncbi_lineage_names"]), axis=1
            )
            ncbi_clade_["ncbi_lineage_ranks"] = ncbi_clade_.apply(
                lambda row: safe_trim_ids(row["ncbi_lineage_ranks"], row["ncbi_lineage_names"]), axis=1
            )

            return gbif_clade_, ncbi_clade_

    raise ValueError(f"No common clade found for '{species_name}' with more than one species.")

