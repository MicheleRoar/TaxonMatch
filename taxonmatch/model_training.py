import random
import sys
import time
import textdistance
import numpy as np
import jellyfish as jf
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
from rapidfuzz import fuzz as rapidfuzz_fuzz
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


CLASSIFIERS = {
    "DummyClassifier": DummyClassifier(strategy='stratified', random_state=0),
    "KNeighborsClassifier": KNeighborsClassifier(3),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
    "AdaBoostClassifier": AdaBoostClassifier(algorithm='SAMME', random_state=0),
    "Perceptron": Perceptron(random_state=0),
    "SVC": SVC(probability=True, random_state=0),
    "MLPClassifier": MLPClassifier(max_iter=500, random_state=0),
    "RandomForestClassifier": RandomForestClassifier(random_state=0),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=0),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0),
}

def generate_positive_set(gbif_dataset, ncbi_dataset, n):
    """
    Generate a set of positive matches by merging GBIF and NCBI datasets based on canonical names.

    Args:
    gbif_dataset (pd.DataFrame): The GBIF dataset.
    ncbi_dataset (pd.DataFrame): The NCBI dataset.
    n (int): The number of samples to generate.

    Returns:
    pd.DataFrame: A DataFrame containing positive matches with taxonomy strings and a match flag.
    """
    
    sys.stdout.write(f'\rGenerating positive set: 0.0%')
    sys.stdout.flush()
    
    # Merge datasets on canonical names to identify positive matches.
    matched_df = gbif_dataset[0].merge(ncbi_dataset[0], left_on='canonicalName', right_on='ncbi_canonicalName', how='inner')
    
    # Identifying duplicates in NCBI
    duplicates = matched_df[matched_df.duplicated(subset='ncbi_canonicalName', keep=False)]
    
    # Identifying species with more than one kingdom classification
    double_cN = matched_df.groupby('canonicalName').filter(lambda x: len(set(x['kingdom'])) > 1)

    # Combine both lists into a set to remove duplicates
    doubles_list = list(set(duplicates.canonicalName) | set(double_cN.canonicalName))

    # Selecting true pairs for classifier, excluding `doubles_list`
    positive_matches = matched_df[~matched_df["canonicalName"].isin(doubles_list)]
    positive_matches = positive_matches[["gbif_taxonomy", "ncbi_target_string", "taxonRank", "ncbi_rank"]]

    # Sampling exact and not-exact matches.
    value = round((n/3) * 2)
    not_exact = positive_matches.query("ncbi_target_string != gbif_taxonomy & not gbif_taxonomy.str.contains('tracheophyta')").sample(value)
    
    progress = (value / n) * 100
    sys.stdout.write(f'\rGenerating positive set: {progress:.1f}%')
    sys.stdout.flush()

    exact = positive_matches.query("ncbi_target_string == gbif_taxonomy").sample(round(n/3))

    progress = 100.0 
    sys.stdout.write(f'\rGenerating positive set: {progress:.1f}%\n')
    sys.stdout.flush()
    
    # Combining samples and marking them as matches.
    positive_matches = pd.concat([exact, not_exact], axis=0)
    positive_matches["match"] = 1
    
    return positive_matches


def generate_negative_set(gbif_dataset, ncbi_dataset, n):
    """
    Generate a set of negative matches for the classifier by sampling and comparing species names.

    Args:
    gbif_dataset (pd.DataFrame): The GBIF dataset.
    ncbi_dataset (pd.DataFrame): The NCBI dataset.
    n (int): The number of negative samples to generate.

    Returns:
    pd.DataFrame: A DataFrame containing negative matches with full taxonomy and a match flag set to 0.
    """

    gbif_samples = list(gbif_dataset[0].canonicalName.str.lower())
    ncbi_samples = list(ncbi_dataset[0].ncbi_canonicalName.str.lower())

    # Remove "unknown" values from the NCBI rank
    ncbi_filtered = ncbi_dataset[0][ncbi_dataset[0].ncbi_rank != "unknown"]

    num_species_samples = round((n * 2) / 3)  # 2/3 species/subspecies
    num_other_samples = round(n / 3)  # 1/3 diffreent hierarchies

    v = []
    total_previous_iterations = 0

    for i, item_a in enumerate(random.sample(gbif_samples, num_species_samples)):
        progress = ((i + total_previous_iterations) / n) * 100
        sys.stdout.write(f'\rGenerating negative set: {progress:.1f}%')
        sys.stdout.flush()

        best_match, similarity = find_most_similar_match(item_a, ncbi_samples)
        v.append((item_a, best_match))

    total_previous_iterations += num_species_samples
    similarity_df = pd.DataFrame(v, columns=['gbif_sample', 'ncbi_sample'])

    temp_df = similarity_df.merge(
        gbif_dataset[0], left_on='gbif_sample',
        right_on=gbif_dataset[0]['canonicalName'].str.lower(),
        how='left'
    ).drop_duplicates("gbif_sample")

    temp_df2 = temp_df.merge(
        ncbi_filtered, left_on='ncbi_sample',
        right_on=ncbi_filtered['ncbi_canonicalName'].str.lower(),
        how='left'
    ).drop_duplicates("ncbi_sample")

    temp_df2 = temp_df2.query("gbif_sample != ncbi_sample")

    false_matches = temp_df2[["gbif_taxonomy", "ncbi_target_string", "taxonRank", "ncbi_rank"]].copy()
    false_matches["match"] = 0

    # Generate pairings for other hierarchies (1/3 of the cases)
    gbif_higher_taxa = gbif_dataset[0][gbif_dataset[0].taxonRank.isin(["family", "order", "class"])]
    ncbi_higher_taxa = ncbi_filtered[ncbi_filtered.ncbi_rank.isin(["family", "order", "class"])]

    higher_gbif_samples = gbif_higher_taxa.sample(num_other_samples)
    higher_ncbi_samples = ncbi_higher_taxa.sample(num_other_samples)

    
    progress = (total_previous_iterations / n) * 100
    sys.stdout.write(f'\rGenerating negative set: {progress:.1f}%')
    sys.stdout.flush()

    higher_negative_matches = pd.DataFrame({
        "gbif_taxonomy": higher_gbif_samples.gbif_taxonomy.values,
        "ncbi_target_string": higher_ncbi_samples.ncbi_target_string.values,
        "taxonRank": higher_gbif_samples.taxonRank.values,
        "ncbi_rank": higher_ncbi_samples.ncbi_rank.values,
        "match": 0
    })

    # Combine all negative matches
    negative_set = pd.concat([false_matches, higher_negative_matches], axis=0).reset_index(drop=True)

    sys.stdout.write(f'\rGenerating negative set: 100.0%\n')
    sys.stdout.flush()

    return negative_set



# Define taxonomic rank hierarchy
rank_hierarchy = {
    "superkingdom": 1, "kingdom": 2, "phylum": 3, "class": 4, "order": 5, 
    "family": 6, "genus": 7, "species": 8, "subspecies": 9, 
    "variety": 9, "form": 9
}

# Distance functions
distances = {
    'levenshtein_distance': textdistance.levenshtein,
    'damerau_levenshtein_distance': textdistance.damerau_levenshtein,
    'hamming_distance': jf.hamming_distance,
    'jaro_similarity': textdistance.jaro,
    'jaro_winkler_similarity': textdistance.jaro_winkler,
}

# Fuzzy matching functions
fuzzy_ratios = {
    'ratio': rapidfuzz_fuzz.ratio,
    'partial_ratio': rapidfuzz_fuzz.partial_ratio,
    'token_sort_ratio': rapidfuzz_fuzz.token_sort_ratio,
    'token_set_ratio': rapidfuzz_fuzz.token_set_ratio,
    'w_ratio': rapidfuzz_fuzz.WRatio,
    'q_ratio': rapidfuzz_fuzz.QRatio
}

# Precompute rank distance similarity mapping
rank_distance_map = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

def compute_rank_similarity(gbif_rank, ncbi_rank):
    gbif_level = rank_hierarchy.get(gbif_rank, 10)  # Evita `.lower()`
    ncbi_level = rank_hierarchy.get(ncbi_rank, 10)
    rank_distance = min(abs(gbif_level - ncbi_level), 5)  # Cap massimo a 5
    return rank_distance_map[rank_distance]

def compute_similarity_metrics(df):
    """Vectorized computation of taxonomic and text similarity metrics."""
    
    df = df.rename(columns={"gbif_taxonomy": "query_name", "ncbi_target_string": "target_name"})
    df["query_name"] = df["query_name"].astype(str).str.lower()
    df["target_name"] = df["target_name"].astype(str).str.lower()

    # Compute rank similarity in batch
    df["rank_similarity"] = np.vectorize(compute_rank_similarity)(
        df["taxonRank"].values, df["ncbi_rank"].values
    )

    # Preallocate distance arrays
    num_samples = len(df)
    all_distances = np.zeros((num_samples, len(distances) + len(fuzzy_ratios)))

    query_names = df["query_name"].values
    target_names = df["target_name"].values

    # Compute distances and fuzzy ratios in batch
    for i, (col_name, func) in enumerate(distances.items()):
        all_distances[:, i] = [func(q, t) for q, t in zip(query_names, target_names)]

    for i, (col_name, func) in enumerate(fuzzy_ratios.items(), start=len(distances)):
        all_distances[:, i] = [func(q, t) for q, t in zip(query_names, target_names)]

    # Assign computed distances back to DataFrame
    all_col_names = list(distances.keys()) + list(fuzzy_ratios.keys())
    for i, col_name in enumerate(all_col_names):
        df[col_name] = all_distances[:, i]

    return df

def prepare_data(positive_matches, negative_matches):
    """
    Optimized version of prepare_data() using batch processing for similarity computation.
    
    Args:
    - positive_matches (pd.DataFrame): Positive matches dataset.
    - negative_matches (pd.DataFrame): Negative matches dataset.

    Returns:
    - pd.DataFrame: Dataset with taxonomic similarity features.
    """
    # Concatenate positive and negative matches
    full_training_set = pd.concat([positive_matches, negative_matches], ignore_index=True)

    # Compute similarity metrics on the entire DataFrame (vectorized)
    df_output = compute_similarity_metrics(full_training_set)

    # Add match column back
    df_output["match"] = full_training_set["match"].values

    return df_output


def get_confusion_matrix_values(y_true, y_pred):
    
    """
    Extract values from a confusion matrix for given test labels and predictions.

    Args:
    y_test (array-like): True labels.
    y_pred (array-like): Predicted labels by the model.

    Returns:
    tuple: A tuple containing the values of the confusion matrix (TP, FP, FN, TN).
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp, fp, fn, tn


def generate_training_test(df_output):
    """
    Split the data into training and testing sets.

    Args:
    df_output (pd.DataFrame): The DataFrame containing the data.

    Returns:
    tuple: A tuple containing training and testing sets (X_train, X_test, y_train, y_test).
    """

    # Extract feature columns, excluding non-feature columns
    feature_columns = [col for col in df_output.columns if col not in ["query_name", "target_name", "taxonRank", "ncbi_rank", "match"]]

    # Extract feature matrix
    X = df_output[feature_columns].values
    y = df_output["match"].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    return X_train, X_test, y_train, y_test


def compare_models(X_train, X_test, y_train, y_test, cv_folds=5, random_seed=42):
    """
    Compare various machine learning models to find the best performing model.

    Args:
    X_train, X_test (array-like): Training and testing feature sets.
    y_train, y_test (array-like): Training and testing labels.

    Returns:
    pd.DataFrame: A DataFrame containing the performance metrics of each model.
    """
    np.random.seed(random_seed)
    results_list = []

    for name, clf in CLASSIFIERS.items():
        start_time = time.time()
        model = clf.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Try to get probability scores for AUC
        try:
            y_score = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            try:
                y_score = model.decision_function(X_test)
            except Exception:
                y_score = None

        roc = roc_auc_score(y_test, y_score) if y_score is not None else np.nan
        run_time = round((time.time() - start_time) / 60, 2)
        tp, fp, fn, tn = get_confusion_matrix_values(y_test, y_pred)

        # Cross-validation score on training set
        try:
            train_cv_accuracy = cross_val_score(clf, X_train, y_train, cv=cv_folds, scoring='accuracy').mean()
        except Exception:
            train_cv_accuracy = np.nan

        results_list.append({
            'model': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'train_cv_accuracy': train_cv_accuracy,
            'mae': mean_absolute_error(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc': roc,
            'run_time_min': run_time,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })

    return pd.DataFrame(results_list).sort_values(['accuracy', 'precision'], ascending=[False, False]).reset_index(drop=True)


def plot_roc_curves(models, X_test, y_test, top_n=5, save_path="roc_curves.png"):
    """
    Plot ROC curves for top N models (must have `predict_proba` or `decision_function`)
    """
    plt.figure(figsize=(8, 6))
    colormap = plt.colormaps['tab10']
    plotted = 0

    for i, (name, model) in enumerate(models.items()):
        try:
            y_score = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            try:
                y_score = model.decision_function(X_test)
            except Exception:
                continue  # Skip models that can't produce scores

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.6f})', color=colormap(i % 10))
        plotted += 1
        if plotted >= top_n:
            break

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves of Top Classifiers')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()




def add_predictions_to_features(features, y_pred):

    """
    Add prediction results to each set of features for analysis.

    Args:
    features (list of dict): List of feature dictionaries.
    y_pred (array-like): Predicted labels by the model.

    Returns:
    None: The function modifies the features list in place.
    """

    for i, prediction in enumerate(y_pred):
        features[i]['prediction'] = prediction

def extract_values_from_dict_list(dict_list):
    """
    Extract feature values from a list of dictionaries for model training.

    Args:
    dict_list (list of dict): List of dictionaries containing features.

    Returns:
    list: A list of lists, each containing the feature values.
    """

    feature_matrix = []
    for comparison in dict_list:
        # Extract all numerical features, skipping the first three fields (query_name, target_name, score)
        values_from_fourth = list(comparison.values())[3:]  
        feature_matrix.append(values_from_fourth)

    return feature_matrix


def finetune_model(model, param_grid, cv, X_train, X_test, y_train, y_test):

    """
    Fine-tune a machine learning model using GridSearchCV.

    Args:
    model (estimator): The machine learning model to fine-tune.
    param_grid (dict): The grid of parameters to search over.
    cv (int): The number of cross-validation folds.
    X_train, X_test (array-like): Training and testing feature sets.
    y_train, y_test (array-like): Training and testing labels.

    Returns:
    dict: A dictionary containing the best parameters found during fine-tuning.
    """

    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv, n_jobs=multiprocessing.cpu_count())
    grid_search.fit(X_train, y_train)
    # Displaying the best parameters and score.
    print("Optimal parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    # Training the model with optimal parameters and evaluating on the test set.
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test set accuracy:", accuracy)
    return grid_search.best_params_


def find_most_similar_match(query, strings):

    """
    Find the most similar string to a given query string from a list of strings, with a similarity threshold.

    Args:
    query (str): The query string to compare.
    strings (list of str): A list of strings to compare against the query.

    Returns:
    tuple: (str, float) - the best matching string and its similarity score.
    """

    # Find the string with the highest similarity to the query string, below 85%.
    best_match = None
    highest_similarity = 0
    
    for string in strings:
        similarity = rapidfuzz_fuzz.ratio(query, string)
        if similarity > highest_similarity and similarity < 85:
            highest_similarity = similarity
            best_match = string
    
    return best_match, highest_similarity