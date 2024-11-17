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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def generate_positive_set(gbif_dataset, ncbi_dataset, n):

    """
    Generate a set of positive matches by merging GBIF and NCBI datasets based on canonical names.

    Args:
    gbif_dataset (pd.DataFrame): The GBIF dataset.
    ncbi_dataset (pd.DataFrame): The NCBI dataset.
    n (int): The number of samples to generate.

    Returns:
    pd.DataFrame: A DataFrame containing positive matches with both taxonomy strings and a match flag.
    """

    # Merge datasets on canonical names to identify positive matches.
    matched_df = gbif_dataset[0].merge(ncbi_dataset[0], left_on='canonicalName', right_on= 'ncbi_canonicalName', how='inner')

    # Identifying duplicates and species with more than one kingdom classification.
    duplicates = matched_df[matched_df.duplicated(subset='ncbi_canonicalName', keep=False)]
    double_cN = matched_df.groupby('canonicalName').filter(lambda x: len(set(x['kingdom'])) > 1)
    doubles_list = list(set(duplicates.canonicalName) | set(double_cN.canonicalName))

    # Selecting true pairs for classifier.
    positive_matches = matched_df[~matched_df["canonicalName"].isin(list(set(double_cN.canonicalName)))]
    positive_matches = positive_matches[["gbif_taxonomy", "ncbi_target_string", "ncbi_rank"]]
    
    # Sampling exact and not-exact matches.
    value = round((n/3) *2)
    not_exact = positive_matches.query("ncbi_target_string != gbif_taxonomy & not gbif_taxonomy.str.contains('tracheophyta')").sample(value)
    exact = positive_matches.query("ncbi_target_string == gbif_taxonomy").sample(round(n/3))
    
    # Combining samples and marking them as matches.
    positive_matches = pd.concat([exact, not_exact], axis=0)
    positive_matches["match"] = 1
    return positive_matches


def find_most_similar_match(query, strings):

    """
    Find the most similar string to a given query string from a list of strings, with a similarity threshold.

    Args:
    query (str): The query string to compare.
    strings (list of str): A list of strings to compare against the query.

    Returns:
    tuple: A tuple containing the best match and its similarity score.
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



def generate_negative_set(gbif_dataset, ncbi_dataset, n):
    
    """
    Generate a set of negative matches for the classifier by sampling and comparing species names.

    Args:
    gbif_dataset (pd.DataFrame): The GBIF dataset.
    ncbi_dataset (pd.DataFrame): The NCBI dataset.
    n (int): The number of negative samples to generate.

    Returns:
    pd.DataFrame: A DataFrame containing negative matches with taxonomy strings and a match flag set to 0.
    """
        
    # Sampling species names from both datasets.
    gbif_samples = list(gbif_dataset[0].canonicalName.str.lower())[1:]
    ncbi_samples = list(ncbi_dataset[0].ncbi_canonicalName.str.lower())

    species = random.sample([item for item in gbif_samples if len(item.split()) >= 2], round(n/3))

    # Finding non-matching pairs.
    list_a = species
    list_b = ncbi_samples

    v = []
    for item_a in range(len(list_a)):
        sys.stdout.write('\r samples {} out of {}'.format(item_a + 1, len(list_a)))
        sys.stdout.flush()

        best_match, similarity = find_most_similar_match(list_a[item_a], list_b)
        v.append((list_a[item_a], best_match, similarity)) 

    # Creating a DataFrame from the matched pairs.
    similarity_df = pd.DataFrame(v, columns=['gbif_sample', 'ncbi_sample', 'similarity score'])

    # Merging with original datasets to get full information.
    temp_df = similarity_df.merge(gbif_dataset[0], left_on='gbif_sample', right_on=gbif_dataset[0]['canonicalName'].str.lower(), how='left').drop_duplicates("gbif_sample")
    temp_df2 = temp_df.merge(ncbi_dataset[0], left_on='ncbi_sample', right_on=ncbi_dataset[0]['ncbi_canonicalName'].str.lower(), how='left').drop_duplicates("ncbi_sample")
    temp_df2 = temp_df2.query("gbif_sample != ncbi_sample")
    false_matches = temp_df2[["gbif_taxonomy", "ncbi_target_string", "ncbi_rank"]].copy()
    false_matches["match"] = 0
    
    # Generating fake matches for subspecies and species.
    subspecies_list = list(gbif_dataset[0][gbif_dataset[0].taxonRank == "subspecies"].sample(round(n/3)).gbif_taxonomy)
    species_list = list(gbif_dataset[0][gbif_dataset[0].taxonRank == "species"].sample(round(n/3)).gbif_taxonomy)
    gbif_fake_list = subspecies_list + species_list
    
    spec = [stringa.rsplit(' ', 1)[0] for stringa in subspecies_list]
    spec_2 = [stringa.rsplit(';', 1)[0] for stringa in species_list]
    ncbi_fake_matches = spec + spec_2
    
    # Combining false and fake matches.
    fake_matches = pd.DataFrame({'gbif_taxonomy': gbif_fake_list, 'ncbi_target_string': ncbi_fake_matches, "ncbi_rank": "subspecies", "match": 0})
    negative_set = pd.concat([false_matches, fake_matches], axis=0)

    return negative_set


def tuple_engineer_features(tuple_list, distance_list=None):
    
    """
    Engineer features by calculating various string distance and similarity measures for given tuples.

    Args:
    tuple_list (list of tuples): List of tuples containing pairs of strings to compare.
    distance_list (list of str, optional): List of distance/similarity measures to apply.

    Returns:
    list: A list of dictionaries containing the engineered features.
    """

    if distance_list is None:
        distance_list = [
            'levenshtein_distance', 'damerau_levenshtein_distance', 'hamming_distance', 
            'jaro_similarity', 'jaro_winkler_similarity', 'ratio', 'partial_ratio',
            'token_sort_ratio', 'token_set_ratio', 'w_ratio', 'q_ratio'
        ]

    # Define distance functions.
    distances = {
        'levenshtein_distance': textdistance.levenshtein,
        'damerau_levenshtein_distance': textdistance.damerau_levenshtein,
        'hamming_distance': jf.hamming_distance,
        'jaro_similarity': textdistance.jaro,
        'jaro_winkler_similarity': textdistance.jaro_winkler,
    }

    # Define fuzzy matching functions.
    fuzzy_ratios = {
        'ratio': rapidfuzz_fuzz.ratio,
        'partial_ratio': rapidfuzz_fuzz.partial_ratio,
        'token_sort_ratio': rapidfuzz_fuzz.token_sort_ratio,
        'token_set_ratio': rapidfuzz_fuzz.token_set_ratio,
        'w_ratio': rapidfuzz_fuzz.WRatio,
        'q_ratio': rapidfuzz_fuzz.QRatio
    }

    result = []

    for tuple_data in tuple_list:
        query_name = str(tuple_data[0]).lower()
        target_name = str(tuple_data[1]).lower()
        score = tuple_data[2]
        feature_row = {'query_name': query_name, 'target_name': target_name, 'score': score}

        for col_name in distance_list:
            if col_name in distances:
                distance_function = distances[col_name]
                feature_row[col_name] = distance_function(query_name, target_name)
            elif col_name in fuzzy_ratios:
                ratio_function = fuzzy_ratios[col_name]
                feature_row[col_name] = ratio_function(query_name, target_name)

        result.append(feature_row)

    return result


def prepare_data(positive_matches, negative_matches):
    
    """
    Prepare the dataset for training by combining positive and negative matches and calculating correlations.

    Args:
    positive_matches (pd.DataFrame): DataFrame of positive matches.
    negative_matches (pd.DataFrame): DataFrame of negative matches.

    Returns:
    tuple: A tuple containing the prepared DataFrame and a list of relevant features.
    """

    # Concatenating positive and negative matches.
    full_training_set = pd.concat([positive_matches, negative_matches], ignore_index=True)
    # Computing similarity measures among samples.
    output = tuple_engineer_features(full_training_set.values.tolist())
    df_output = pd.DataFrame(output)
    df_output["match"] = full_training_set["match"]
    # Correlation with output variable.
    cor_target = abs(pd.DataFrame(df_output).corr(numeric_only=True)["match"])
    relevant_features = list(pd.DataFrame(cor_target).sort_values("match", ascending=False)[1:].index)
    return df_output


def get_confusion_matrix_values(y_test, y_pred):
    
    """
    Extract values from a confusion matrix for given test labels and predictions.

    Args:
    y_test (array-like): True labels.
    y_pred (array-like): Predicted labels by the model.

    Returns:
    tuple: A tuple containing the values of the confusion matrix (TP, FP, FN, TN).
    """

    cm = confusion_matrix(y_test, y_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])


def generate_training_test(df_output):
    
    """
    Split the data into training and testing sets.

    Args:
    df_output (pd.DataFrame): The DataFrame containing the data.
    relevant_features (list of str): List of relevant features for training.

    Returns:
    tuple: A tuple containing training and testing sets (X_train, X_test, y_train, y_test).
    """

    relevant_features=['levenshtein_distance', 'damerau_levenshtein_distance', 'ratio', 'q_ratio', 'token_sort_ratio', 'w_ratio', 'token_set_ratio', 'jaro_winkler_similarity', 'partial_ratio', 'hamming_distance', 'jaro_similarity']


    X = df_output[relevant_features].values
    y = df_output['match'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test

def compare_models(X_train, X_test, y_train, y_test):    

    """
    Compare various machine learning models to find the best performing model.

    Args:
    X_train, X_test (array-like): Training and testing feature sets.
    y_train, y_test (array-like): Training and testing labels.

    Returns:
    pd.DataFrame: A DataFrame containing the performance metrics of each model.
    """

    classifiers = {
        # A collection of different classifiers from scikit-learn library.
        "DummyClassifier": DummyClassifier(strategy='stratified', random_state=0),
        "KNeighborsClassifier": KNeighborsClassifier(3),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "AdaBoostClassifier": AdaBoostClassifier(algorithm='SAMME'),
        "Perceptron": Perceptron(),
        "SVC": SVC(),
        "MLPClassifier": MLPClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "XGBClassifier": XGBClassifier(),
    }
    results_list = []

    # DataFrame to store the results of each classifier.
    for key in classifiers:
        # Training each classifier and calculating performance metrics.
        start_time = time.time()
        classifier = classifiers[key]
        model = classifier.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculating various metrics like mean absolute error, accuracy, precision, etc.
        mae = mean_absolute_error(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_pred)
        run_time = format(round((time.time() - start_time)/60,2))
        tp, fp, fn, tn = get_confusion_matrix_values(y_test, y_pred)

        # Adding the calculated metrics for each model to the DataFrame.
        row = {'model': key, 'accuracy': accuracy, 'mae': mae, 'precision': precision, 'recall': recall, 'f1': f1, 'roc': roc, 'run_time': run_time, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        results_list.append(row)

    df_results = pd.DataFrame(results_list)
    
    # Returning the sorted results based on accuracy and precision.
    return df_results.sort_values(['accuracy', 'precision'], ascending=[False, False]).reset_index(drop = True)

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
        # Skipping the first three elements in the dictionary and taking the rest.
        values_from_third = list(comparison.values())[3:] 
        feature_matrix.append(values_from_third)
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