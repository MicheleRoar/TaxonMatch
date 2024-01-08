import os
import csv
import gzip
import zipfile
import subprocess
import pandas as pd
from tqdm import tqdm
import urllib.request



def download_gbif(output_folder=None):
    
    if output_folder is None:
        output_folder = os.getcwd()

    url = "https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip"
    filename = os.path.join(output_folder, "backbone.zip")  

    # Use tqdm to display a progress bar
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading GBIF Taxonomic Data") as t:
        urllib.request.urlretrieve(url, filename, reporthook=lambda blocknum, blocksize, totalsize: t.update(blocksize))

    if os.path.exists(filename):
        print("GBIF backbone taxonomy has been downloaded successfully.")

    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extract('Taxon.tsv', output_folder)

    tsv_path = os.path.join(output_folder, 'Taxon.tsv')
    if os.path.exists(tsv_path):
        #print(f"Taxon.tsv has been extracted and stored at {tsv_path}")
        os.remove(filename)
        #return(tsv_path)
    else:
        raise RuntimeError("Error saving Taxon.tsv. Consider raising an issue on Github.")




def compare_versions(version1, version2):
    v1_components = list(map(int, version1.split(".")))
    v2_components = list(map(int, version2.split(".")))

    for v1, v2 in zip(v1_components, v2_components):
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1

    return 0




def download_ncbi(taxonkitpath=None, output_folder=None):
    if taxonkitpath is not None:
        if "taxonkit" not in taxonkitpath.lower():
            raise ValueError("The path must include both the directory name and filename 'taxonkit'")
        try:
            version_output = subprocess.check_output([taxonkitpath, "version"], stderr=subprocess.DEVNULL, text=True)
            version = version_output.strip().replace("taxonkit v", "")
            if compare_versions(version, "0.8.0") == 0:
                raise ValueError("Taxonkit version 0.8.0 or greater is required. Please download a more recent version of Taxonkit and try again.")
            #else:
                #print(f"Taxonkit v{version} detected!")
        except subprocess.CalledProcessError:
            raise ValueError(f"Taxonkit not detected. Is this the correct path to Taxonkit: {taxonkitpath}?")
    else:
        raise ValueError("Please specify the taxonkit path!")
        return  # This will exit the function

    if output_folder is None:
        output_folder = os.getcwd()

    # Create a new folder for Taxonkit output
    taxonkit_output_folder = os.path.join(output_folder, "NCBI_output")
    os.makedirs(taxonkit_output_folder, exist_ok=True)

    # Download the NCBI taxondump
    url = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdmp.zip"
    tf = os.path.join(taxonkit_output_folder, "taxdmp.zip")
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading NCBI Taxonomic Data") as pbar:
        def report_hook(blocknum, blocksize, totalsize):
            pbar.update(blocknum * blocksize / totalsize * 100)
        urllib.request.urlretrieve(url, tf, reporthook=report_hook)


    # Decompressing directly into the new folder
    with zipfile.ZipFile(tf, "r") as zip_ref:
        zip_ref.extractall(taxonkit_output_folder)

    gz_path = taxonkit_output_folder + '/All.lineages.tsv.gz'
    file_path = taxonkit_output_folder + '/ncbi_data.tsv'

    # Constructing the combined command
    full_command = (
        f"{taxonkitpath} --data-dir={taxonkit_output_folder} list --ids 1 | "
        f"{taxonkitpath} lineage --show-lineage-taxids --show-lineage-ranks --show-rank --show-name --data-dir={taxonkit_output_folder} | "
        f"{taxonkitpath} reformat --taxid-field 1 --data-dir={taxonkit_output_folder} -o {gz_path}"
    )

    # Running the combined command without printing the output
    try:
        subprocess.run(full_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("NCBI taxonomic data has been downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print("Error:", e)


    # Apre il file compresso in modalità lettura binaria ('rb')
    with gzip.open(gz_path, 'rb') as gz_file:
        # Decodifica il contenuto decompresso in modalità lettura testuale ('rt')
        decompressed_content = gz_file.read().decode('utf-8')

    # Apre il file di output in modalità scrittura testuale ('wt')
    with open(file_path, 'wt') as output_file:
        # Crea un oggetto CSV writer
        csv_writer = csv.writer(output_file, delimiter='\t')

        # Legge le righe dal contenuto decompresso e scrive nel file di output
        for line in decompressed_content.splitlines():
            row = line.split('\t')  # Divide la riga in campi delimitati da tabulazioni
            csv_writer.writerow(row)


################################################################################################

import re

def create_gbif_taxonomy(row):
    if row['taxonRank'] in ['species', 'subspecies', "variety"]:
        return f"{row['phylum']};{row['class']};{row['order']};{row['family']};{row['genus']};{row['canonicalName']}".lower()
    else:
        return f"{row['phylum']};{row['class']};{row['order']};{row['family']};{row['genus']}".lower()

def find_all_parents(taxon_id, parents_dict):
    parents = []
    while taxon_id != -1:  # -1 represents the NaN converted value
        parents.append(taxon_id)
        taxon_id = parents_dict.get(taxon_id, -1)  # Get the parent

    # Reverse the list of parents and create a string separated by ";", excluding the first element
    parents_string = ';'.join(map(str, parents[::-1][1:])) if len(parents) > 1 else ''

    return parents_string

def load_gbif_samples(gbif_path_file):

    columns_of_interest = ['taxonID', 'parentNameUsageID', 'acceptedNameUsageID', 'canonicalName', 'taxonRank', 'taxonomicStatus', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    gbif_full = pd.read_csv(gbif_path_file, sep="\t", usecols=columns_of_interest, on_bad_lines='skip', low_memory=False)

    #| taxonomicStatus == 'synonym')
    gbif_subset = gbif_full.query("taxonomicStatus == 'accepted' & taxonRank != 'unranked'").fillna('').drop_duplicates(subset=gbif_full.columns[1:], keep='first')

    gbif_subset['gbif_taxonomy'] = gbif_subset.apply(create_gbif_taxonomy, axis=1)
    gbif_subset['gbif_taxonomy'] = gbif_subset['gbif_taxonomy'].str.replace(r'(\W)\1+', r'\1', regex = True)

    # Rimuovi i ';' dalla fine di ogni stringa e sostituiscili con una stringa vuota
    gbif_subset['gbif_taxonomy'] = gbif_subset['gbif_taxonomy'].str.rstrip(';')

    # Rimuovi i ';' dall'inizio di ogni stringa
    gbif_subset['gbif_taxonomy'] = gbif_subset['gbif_taxonomy'].str.lstrip(';')

    # Sostituisci le celle contenenti solo ';' con una stringa vuota
    gbif_subset.loc[gbif_subset['gbif_taxonomy'] == ';', 'gbif_taxonomy'] = ''
    gbif_subset = gbif_subset.drop_duplicates(subset="gbif_taxonomy")

    gbif_subset['parentNameUsageID'] = gbif_subset['parentNameUsageID'].replace('', np.nan).fillna(-1).astype(int)  # Convert to int, handling NaNs
    parents_dict = dict(zip(gbif_subset['taxonID'], gbif_subset['parentNameUsageID']))
    gbif_subset['gbif_taxonomy_ids'] = gbif_subset.apply(lambda x: find_all_parents(x['taxonID'], parents_dict), axis=1)
    
    return gbif_subset, gbif_full


def prepare_ncbi_strings(row):
    parts = row['ncbi_target_string'].split(';')
    
    if row['ncbi_rank'] in ['species', 'subspecies', 'strain']:
        new_string = ';'.join(parts[1:-1]) + ';' + row['ncbi_canonicalName']
    else:
        new_string = ';'.join(parts[1:-1])
    
    return new_string.lower()


def remove_extra_separators(s):
    return re.sub(r';+', ';', s)



def load_ncbi_samples(ncbi_path_file):

    ncbi_full = pd.read_csv(ncbi_path_file, sep="\t", names=['ncbi_id', 'ncbi_lineage_names', 'ncbi_lineage_ids', 'ncbi_canonicalName', 'ncbi_rank', 'ncbi_lineage_ranks', 'ncbi_target_string'])
    ncbi_subset = ncbi_full.copy()
    ncbi_subset['ncbi_target_string'] = ncbi_subset.apply(prepare_ncbi_strings, axis=1)
    ncbi_subset["ncbi_target_string"] = ncbi_subset["ncbi_target_string"].apply(remove_extra_separators)
    ncbi_subset["ncbi_target_string"] = ncbi_subset["ncbi_target_string"].str.lower().str.replace(r'^;+|;+$', '', regex=True)
    ncbi_subset = ncbi_subset.drop_duplicates(subset="ncbi_target_string")
    
    return ncbi_subset, ncbi_full





####################################################################################################

import gensim
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import Levenshtein


def get_gbif_synonyms(gbif_dataset):
    
    df_gbif_synonyms = gbif_dataset[1][gbif_dataset[1].taxonomicStatus == 'synonym']
    
    # Crea il dizionario
    
    gbif_synonyms_names = {}
    gbif_synonyms_ids = {}
    
    for index, row in df_gbif_synonyms.iterrows():
        accepted_id = row['acceptedNameUsageID']
        canonical_name = row['canonicalName']
        canonical_name_id = row['taxonID']
        
        # Verifica se il canonical_name non è nullo prima di aggiungerlo al dizionario
        if not pd.isnull(canonical_name):
            if accepted_id in gbif_synonyms_names:
                # Se l'accepted_id è già presente nel dizionario, aggiungi il canonical_name
                gbif_synonyms_names[accepted_id].append(canonical_name)
                gbif_synonyms_ids[accepted_id].append(canonical_name_id)
            else:
                # Se l'accepted_id non è ancora nel dizionario, crea una nuova lista con il canonical_name
                gbif_synonyms_names[accepted_id] = [canonical_name]
                gbif_synonyms_ids[accepted_id] = [canonical_name_id]
    return gbif_synonyms_names,  gbif_synonyms_ids

def get_ncbi_synonyms(names_path):

    # Crea un dizionario che associa i nomi (compresi i sinonimi) a ciascun TaxID
    names_dict = {}
    
    # Lettura di names.dmp
    with open(names_path, 'r') as names_file:
        for line in names_file:
            parts = line.strip().split('|')
            taxid = int(parts[0].strip())
            name = parts[1].strip()
            name_type = parts[3].strip()  # Type of name, e.g., synonym, equivalent name, scientific name
    
            if name_type in ['acronym', 'blast name', 'common name', 'equivalent name', 'genbank acronym', 'genbank common name', 'synonym']:
                if taxid not in names_dict:
                    names_dict[taxid] = []
                names_dict[taxid].append((name))
    return(names_dict)


def get_wordcloud(full_training_set):
    texts = ''
    for index, item in full_training_set.sample(5000).iterrows():
        texts = texts + ' ' + item['gbif_name'] + item['ncbi_name']
        word_cloud = WordCloud(collocations = False, background_color = 'white').generate(texts)
        # plot the WordCloud image
    plt.figure(figsize = (5, 10), facecolor = None)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()


def filter_synonyms(df_matched, df_unmatched, ncbi_synonyms, gbif_synonyms, gbif_dataset, ncbi_dataset):
    
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
        
        # Verifica se il taxonID è nei sinonimi di ncbi_id
        if gbif_canonicalName in ncbi_synonyms.get(ncbi_id, []):
            matching_synonims.append(row)
        
        # Verifica se l'ncbi_id è nei sinonimi di taxonID
        elif ncbi_canonicalName in gbif_synonyms[0].get(taxonID, []):
            matching_synonims.append(row)
        
        # Verifica se c'è un match tra i sinonimi di gbif e ncbi
        for gbif_synonym in gbif_synonyms[0].get(taxonID, []):
            for ncbi_synonym in ncbi_synonyms.get(ncbi_id, []):
                if gbif_synonym == ncbi_synonym:
                    matching_synonims.append(row)
        
        # Se non c'è corrispondenza, aggiungi la riga ai dati esclusi
        if not any(row.equals(existing_row) for existing_row in matching_synonims):
            excluded_data.append(row)
    
    
    # Crea DataFrame separati per i dati inclusi ed esclusi
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


#######################################################################################################


import random
import sys
from rapidfuzz import fuzz as rapidfuzz_fuzz
import textdistance
import numpy as np
import jellyfish as jf


def generate_positive_set(gbif_dataset, ncbi_dataset, n):
    matched_df = gbif_dataset[0].merge(ncbi_dataset[0], left_on='canonicalName', right_on= 'ncbi_canonicalName', how='inner')

    duplicates = matched_df[matched_df.duplicated(subset='ncbi_canonicalName', keep=False)]
    double_cN = matched_df.groupby('canonicalName').filter(lambda x: len(set(x['kingdom'])) > 1)
    doubles_list = list(set(duplicates.canonicalName) | set(double_cN.canonicalName))

    #generating true pairs for the classifier
    positive_matches = matched_df[~matched_df["canonicalName"].isin(list(set(double_cN.canonicalName)))]
    positive_matches = positive_matches[["gbif_taxonomy", "ncbi_target_string", "ncbi_rank"]]
    
    value = round((n/3) *2)
    not_exact = positive_matches.query("ncbi_target_string != gbif_taxonomy & not gbif_taxonomy.str.contains('tracheophyta')").sample(value)
    exact = positive_matches.query("ncbi_target_string == gbif_taxonomy").sample(round(n/3))
    
    positive_matches = pd.concat([exact, not_exact], axis=0)
    positive_matches["match"] = 1
    return positive_matches


def find_most_similar_match(query, strings):
    best_match = None
    highest_similarity = 0
    
    for string in strings:
        similarity = rapidfuzz_fuzz.ratio(query, string)
        if similarity > highest_similarity and similarity < 85:
            highest_similarity = similarity
            best_match = string
    
    return best_match, highest_similarity



def generate_negative_set(gbif_dataset, ncbi_dataset, n):
        
    gbif_samples = list(gbif_dataset[0].canonicalName.str.lower())[1:]
    ncbi_samples = list(ncbi_dataset[0].ncbi_canonicalName.str.lower())

    species = random.sample([item for item in gbif_samples if len(item.split()) >= 2], round(n/3))

    list_a = species
    list_b = ncbi_samples

    v = []
    for item_a in range(len(list_a)):

        sys.stdout.write('\r samples {} out of {}'.format(item_a + 1, len(list_a)))
        sys.stdout.flush()

        best_match, similarity = find_most_similar_match(list_a[item_a], list_b)
        v.append((list_a[item_a], best_match, similarity)) 

    similarity_df = pd.DataFrame(v, columns=['gbif_sample', 'ncbi_sample', 'similarity score'])

    temp_df = similarity_df.merge(gbif_dataset[0], left_on='gbif_sample', right_on=gbif_dataset[0]['canonicalName'].str.lower(), how='left').drop_duplicates("gbif_sample")
    temp_df2 = temp_df.merge(ncbi_dataset[0], left_on='ncbi_sample', right_on=ncbi_dataset[0]['ncbi_canonicalName'].str.lower(), how='left').drop_duplicates("ncbi_sample")
    temp_df2 = temp_df2.query("gbif_sample != ncbi_sample")
    false_matches = temp_df2[["gbif_taxonomy", "ncbi_target_string", "ncbi_rank"]].copy()
    false_matches["match"] = 0
    
    subspecies_list = list(gbif_dataset[0][gbif_dataset[0].taxonRank == "subspecies"].sample(round(n/3)).gbif_taxonomy)
    species_list = list(gbif_dataset[0][gbif_dataset[0].taxonRank == "species"].sample(round(n/3)).gbif_taxonomy)
    gbif_fake_list = subspecies_list + species_list
    
    spec = [stringa.rsplit(' ', 1)[0] for stringa in subspecies_list]
    spec_2 = [stringa.rsplit(';', 1)[0] for stringa in species_list]
    ncbi_fake_matches = spec + spec_2
    
    fake_matches = pd.DataFrame({'gbif_taxonomy': gbif_fake_list, 'ncbi_target_string': ncbi_fake_matches, "ncbi_rank": "subspecies", "match": 0})
    negative_set = pd.concat([false_matches, fake_matches], axis=0)

    return negative_set



def tuple_engineer_features(tuple_list, distance_list=None):
    if distance_list is None:
        distance_list = [
            'levenshtein_distance', 'damerau_levenshtein_distance', 'hamming_distance', 
            'jaro_similarity', 'jaro_winkler_similarity', 'ratio', 'partial_ratio',
            'token_sort_ratio', 'token_set_ratio', 'w_ratio', 'q_ratio'
        ]

    distances = {
        'levenshtein_distance': textdistance.levenshtein,
        'damerau_levenshtein_distance': textdistance.damerau_levenshtein,
        'hamming_distance': jf.hamming_distance,
        'jaro_similarity': textdistance.jaro,
        'jaro_winkler_similarity': textdistance.jaro_winkler,
    }

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
        gbif_name = str(tuple_data[0]).lower()
        ncbi_name = str(tuple_data[1]).lower()
        score = tuple_data[2]
        feature_row = {'gbif_name': gbif_name, 'ncbi_name': ncbi_name, 'score': score}

        for col_name in distance_list:
            if col_name in distances:
                distance_function = distances[col_name]
                feature_row[col_name] = distance_function(gbif_name, ncbi_name)
            elif col_name in fuzzy_ratios:
                ratio_function = fuzzy_ratios[col_name]
                feature_row[col_name] = ratio_function(gbif_name, ncbi_name)

        result.append(feature_row)

    return result



def prepare_data(positive_matches, negative_matches):
    #generating training for the classifier
    full_training_set = pd.concat([positive_matches, negative_matches], ignore_index=True)
    #computing similarity measures among samples
    output = tuple_engineer_features(full_training_set.values.tolist())
    df_output = pd.DataFrame(output)
    df_output["match"] = full_training_set["match"]
    #Correlation with output variable
    cor_target = abs(pd.DataFrame(df_output).corr(numeric_only=True)["match"])
    relevant_features = list(pd.DataFrame(cor_target).sort_values("match", ascending=False)[1:].index)
    return df_output, relevant_features


################################################################################################


import time
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
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


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import multiprocessing



def get_confusion_matrix_values(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])


def generate_training_test(df_output, relevant_features):
    #creating training and testing data
    X = df_output[relevant_features].values
    y = df_output['match'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test

def compare_models(X_train, X_test, y_train, y_test):    
    #selecting and comparing state of the art classifiers to choose the best one
    classifiers = {
        "DummyClassifier_stratified":DummyClassifier(strategy='stratified', random_state=0),    
        "KNeighborsClassifier":KNeighborsClassifier(3),
        "DecisionTreeClassifier":DecisionTreeClassifier(),
        "AdaBoostClassifier":AdaBoostClassifier(),
        "Perceptron": Perceptron(),
        "SupportVectorMachine":SVC(),
        "MLP": MLPClassifier(),
        "RandomForestClassifier":RandomForestClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "XGBClassifier":XGBClassifier(),
    }

    df_results = pd.DataFrame(columns=['model', 'accuracy', 'mae', 'precision','recall','f1','roc','run_time','tp','fp','tn','fn'])
    for key in classifiers:

        start_time = time.time()
        classifier = classifiers[key]
        model = classifier.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_pred)
        classification = classification_report(y_test, y_pred, zero_division=0)
        run_time = format(round((time.time() - start_time)/60,2))
        tp, fp, fn, tn = get_confusion_matrix_values(y_test, y_pred)

        row = {'model': key,
               'accuracy': accuracy,
               'mae': mae,
               'precision': precision,
               'recall': recall,
               'f1': f1,
               'roc': roc,
               'run_time': run_time,
               'tp': tp,
               'fp': fp,
               'tn': tn,
               'fn': fn,
              }

        df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)

    return df_results.sort_values(['accuracy', 'precision'], ascending=[False, False]).reset_index(drop = True)



def add_predictions_to_features(features, y_pred):
    for i, prediction in enumerate(y_pred):
        features[i]['prediction'] = prediction



def extract_values_from_dict_list(dict_list):
    feature_matrix = []

    for comparison in dict_list:
        values_from_third = list(comparison.values())[3:] 
        feature_matrix.append(values_from_third)
    
    return feature_matrix


def ngrams(string, n=4):
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower() #make lower case
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string) #remove the list of chars defined above
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single space
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def finetune_model(model, param_grid, cv, X_train, X_test, y_train, y_test):
    
    # Crea l'oggetto GridSearchCV
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5, n_jobs=multiprocessing.cpu_count())

    # Esegui la ricerca dei parametri ottimali
    grid_search.fit(X_train, y_train)

    # Stampa i parametri ottimali e il punteggio migliore
    print("Parametri ottimali:", grid_search.best_params_)
    print("Miglior punteggio:", grid_search.best_score_)

    # Addestra il modello con i parametri ottimali
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Valuta il modello sul set di test
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy sul set di test:", accuracy)
    return grid_search.best_params_


#############################################################################################################


import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors



def match_datasets(query_dataset, target_dataset, model, relevant_features, threshold):

    target = list(set(target_dataset.ncbi_target_string))
    query = list(set(query_dataset.gbif_taxonomy))
    
    vectorizer = TfidfVectorizer(analyzer=ngrams, lowercase=True)
    tfidf = vectorizer.fit_transform(target)

    # Il tuo codice originale
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
    new_df_matched = pd.concat([df_matched, ncbi_missing_2], ignore_index=True)
    new_df_matched = new_df_matched.fillna(-1)

    
    df_unmatched = query_dataset[query_dataset["gbif_taxonomy"].isin(discarded)]
    return (new_df_matched, df_unmatched)


###########################################################################################

import ete3
from anytree import Node, RenderTree


def print_tree(tree):
    # Stampa dell'albero con ID
    for pre, fill, node in RenderTree(tree):
        ncbi_id = getattr(node, 'ncbi_id', None)
        gbif_id = getattr(node, 'gbif_taxon_id', None)
        id_info = f" (NCBI ID: {ncbi_id}, GBIF ID: {gbif_id})" if ncbi_id or gbif_id else ""
        print(f"{pre}{node.name}{id_info}")




def save_tree(tree, path):
    # Apri un file in modalità scrittura
    with open(path, 'w') as f:
        # Stampa l'albero con ID nel file
        for pre, fill, node in RenderTree(tree):
            ncbi_id = getattr(node, 'ncbi_id', None)
            gbif_id = getattr(node, 'gbif_taxon_id', None)
            id_info = f" (NCBI ID: {ncbi_id}, GBIF ID: {gbif_id})" if ncbi_id or gbif_id else ""
            f.write(f"{pre}{node.name}{id_info}\n")

    print(f"The tree is saved in the file: {path}.")



# Funzione per indicizzare l'albero nel formato richiesto (1.1.1.1.1, 1.1.1.1.2, ecc.)
def index_tree(node, index_list=None, include_name=True):
    if index_list is None:
        index_list = []

    # Genera l'etichetta dell'indice solo per i nodi diversi dalla radice se include_name è True
    if len(index_list) > 0 or (len(index_list) == 0 and include_name):
        node_name = "" if include_name else node.name  # Modificato per escludere il nome del nodo se include_name è False
        node.name = '.'.join(str(i) for i in index_list) + " " + node_name

    for i, child in enumerate(node.children):
        index_list.append(i + 1)  # Aggiungi l'indice del figlio corrente
        index_tree(child, index_list)  # Ricorsione per il figlio
        index_list.pop()  # Rimuovi l'indice del figlio corrente
    return node


# Funzione per generare il DataFrame con NCBI_ID, GBIF_TAXON_ID e l'indicizzazione
def generate_dataframe(node):
    data = []
    for pre, fill, node in RenderTree(node):
        ncbi_id = getattr(node, 'ncbi_id', '')
        gbif_id = getattr(node, 'gbif_taxon_id', '')
        data.append((ncbi_id, gbif_id, node.name))

    df = pd.DataFrame(data, columns=['ncbi_id', 'gbif_taxon_id', 'Index'])
    return df


# Estrarre la stringa dalla parola "Arthropoda" in poi
def extract_after_arthropoda(lineage_names):
    return "arthropoda;" + lineage_names.split("Arthropoda" + ';', 1)[-1].lower()


def select_taxonomic_clade(clade, gbif_dataset, ncbi_dataset):

    # Filtrare e estrarre le righe di interesse dal DataFrame gbif_dataset
    gbif_arthropoda = gbif_dataset[0][gbif_dataset[0].phylum == "Arthropoda"]

    # Filtrare e applicare la funzione di estrazione al DataFrame ncbi_arthropoda
    ncbi_arthropoda = ncbi_dataset[0][ncbi_dataset[0]["ncbi_target_string"].str.contains("Arthropoda".lower() + ";")]
    ncbi_arthropoda_ = ncbi_arthropoda.copy()
    ncbi_arthropoda_.loc[:, 'ncbi_lineage_names'] = ncbi_arthropoda_['ncbi_lineage_names'].apply(extract_after_arthropoda)

    # Filtrare il DataFrame per escludere le righe che contengono numeri in ncbi_lineage_names
    #ncbi_arthropoda_ = ncbi_arthropoda_[~ncbi_arthropoda_['ncbi_lineage_names'].str.contains(r'\d')]

    # Aggiornare ncbi_lineage_ids con lo stesso numero di elementi della nuova ncbi_lineage_names
    ncbi_arthropoda_['ncbi_lineage_ids'] = ncbi_arthropoda_.apply(lambda row: ';'.join(row['ncbi_lineage_ids'].split(';')[-len(row['ncbi_lineage_names'].split(';')):]), axis=1)

    ncbi_arthropoda = ncbi_arthropoda_.iloc[1:]
    return ncbi_arthropoda, gbif_arthropoda



def export_tree(tree, ncbi_dataset, gbif_dataset, path):
    
    # Generate DataFrame
    df_final = generate_dataframe(tree)

    # elimina la prima riga vuota
    df_final = df_final.tail(-1)

    # Converti le colonne in int, trattando i valori vuoti come -1
    df_final['ncbi_id'] = pd.to_numeric(df_final['ncbi_id'], errors='coerce').fillna(-1).astype('int64')
    df_final['gbif_taxon_id'] = pd.to_numeric(df_final['gbif_taxon_id'], errors='coerce').fillna(-1).astype('int64')

    # Effettua il merge basato sulla colonna "ncbi_id" di test con "ncbi_id" di ncbi_dataset
    merged_df1 = pd.merge(df_final, ncbi_dataset[1], left_on='ncbi_id', right_on='ncbi_id', how='left')

    # Effettua il merge basato sulla colonna "gbif_taxon_id" di test con "lataxonID" di gbif_dataset
    final_dataset = pd.merge(merged_df1, gbif_dataset[1], left_on='gbif_taxon_id', right_on='taxonID', how='left')
    
    #Get synonyms from GBIF
    synonym_dict_names, synonym_dict_ids = get_gbif_synonyms(gbif_dataset[1])
    
    # Crea la nuova colonna nel dataframe utilizzando il dizionario
    final_dataset['synonyms_names'] = final_dataset['taxonID'].map(lambda x: '; '.join(map(str, synonym_dict_names.get(x, []))).lower())
    final_dataset['synonyms_ids'] = final_dataset['taxonID'].map(lambda x: '; '.join(map(str, synonym_dict_ids.get(x, []))).lower())
    
    final_dataset[["ncbi_id", "gbif_taxon_id", "Index", "ncbi_canonicalName", "canonicalName", "synonyms_names", 'synonyms_ids']]
    
    # Save DataFrame in a CSV file
    final_dataset.to_csv(path, index=False)
    
    return final_dataset


def tree_to_dataframe(tree):
    data = []
    for node in tree.traverse():
        node_data = {
            "Index": node.name,
            "ncbi_id": node.ncbi_id if "ncbi_id" in node.features else None,
            "gbif_taxon_id": node.gbif_taxon_id if "gbif_taxon_id" in node.features else None,
        }
        data.append(node_data)
    return data


def manage_duplicated_branches(df_matched_2, df_unmatched_2):
    
    common = df_matched_2.query("canonicalName != ncbi_canonicalName")
    pattern = r'^\b\w+\b$'
    filtered_df = common[common['canonicalName'].str.match(pattern, na=False)]
    gbif_to_ncbi = dict(zip(filtered_df['canonicalName'].str.lower(), filtered_df['ncbi_canonicalName'].str.lower()))

    corrected_df_unmatched_2 = df_unmatched_2.copy()

    # Converti la colonna 'gbif_taxonomy' in stringhe
    corrected_df_unmatched_2['gbif_taxonomy'] = corrected_df_unmatched_2['gbif_taxonomy'].astype(str)
    
    # Funzione per sostituire le sottostringhe
    def replace_with_dict(text):
        for key, value in gbif_to_ncbi.items():
            text = re.sub(r'\b{}\b'.format(re.escape(key)), value, text)
        return text
    
    # Applica la sostituzione alla colonna 'gbif_taxonomy'
    corrected_df_unmatched_2['gbif_taxonomy_corrected'] = corrected_df_unmatched_2['gbif_taxonomy'].apply(replace_with_dict)
    return df_matched_2, corrected_df_unmatched_2


def generate_tree(df_matched_2, df_unmatched_2):
    # Creazione dell'albero filogenetico iniziale
    tree = ete3.Tree()

    # Dizionario per tenere traccia dei nodi in base al nome del taxon
    node_dict = {}

    # Dizionario per tenere traccia degli identificatori in base al taxon ID
    taxon_id_dict = {}

    # Inserimento delle informazioni da df_ncbi
    for row in df_matched_2.itertuples(index=False):
        ncbi_lineage_names = row.ncbi_lineage_names.lower().split(';')
        ncbi_lineage_ids = row.ncbi_lineage_ids.split(';')
        ncbi_taxon_id = row.taxonID
    
        parent_node = tree
        for i in range(len(ncbi_lineage_names)):
            taxon = ncbi_lineage_names[i]
            ncbi_id = ncbi_lineage_ids[i]
    
            existing_child = node_dict.get(taxon)
            if not existing_child:
                new_node = parent_node.add_child(name=taxon)
                node_dict[taxon] = new_node
                parent_node = new_node
            else:
                parent_node = existing_child
    
            parent_node.add_feature('ncbi_taxon_id', ncbi_taxon_id)
            parent_node.add_feature('ncbi_id', ncbi_id)
            taxon_id_dict[ncbi_taxon_id] = ncbi_id
    
            # Aggiungi l'id di GBIF come l'id del taxonID per le foglie
            if i == len(ncbi_lineage_names) - 1:
                parent_node.add_feature('gbif_taxon_id', ncbi_taxon_id)
    
    # Inserimento delle informazioni da df_gbif
    for row in df_unmatched_2.itertuples(index=False):
        gbif_taxonomy = row.gbif_taxonomy_corrected.split(';')
        gbif_taxonomy_ids = row.gbif_taxonomy_ids.split(';')
    
        # Verifica se il numero di elementi è lo stesso
        if len(gbif_taxonomy) != len(gbif_taxonomy_ids):
            continue
    
        parent_node = tree
    
        for i in range(len(gbif_taxonomy)):
            taxon = gbif_taxonomy[i]
            gbif_id = gbif_taxonomy_ids[i]
    
            existing_child = node_dict.get(taxon)
            if not existing_child:
                new_node = parent_node.add_child(name=taxon)
                node_dict[taxon] = new_node
                parent_node = new_node
            else:
                parent_node = existing_child
    
            parent_node.add_feature('gbif_taxon_id', gbif_id)
    return tree


def select_taxonomic_clade_2(clade, gbif_dataset, ncbi_dataset):

    # Filtrare e estrarre le righe di interesse dal DataFrame gbif_dataset
    gbif_arthropoda = gbif_dataset[0][gbif_dataset[0].phylum == "Arthropoda"]

    # Filtrare e applicare la funzione di estrazione al DataFrame ncbi_arthropoda
    ncbi_arthropoda = ncbi_dataset[0][ncbi_dataset[0]["ncbi_target_string"].str.contains("Arthropoda".lower() + ";")]
    ncbi_arthropoda_ = ncbi_arthropoda.copy()
    ncbi_arthropoda_.loc[:, 'ncbi_lineage_names'] = ncbi_arthropoda_['ncbi_lineage_names'].apply(extract_after_arthropoda)

    # Filtrare il DataFrame per escludere le righe che contengono numeri in ncbi_lineage_names
    #ncbi_arthropoda_ = ncbi_arthropoda_[~ncbi_arthropoda_['ncbi_canonicalName'].str.contains(r'\d')]

    # Aggiornare ncbi_lineage_ids con lo stesso numero di elementi della nuova ncbi_lineage_names
    ncbi_arthropoda_['ncbi_lineage_ids'] = ncbi_arthropoda_.apply(lambda row: ';'.join(row['ncbi_lineage_ids'].split(';')[-len(row['ncbi_lineage_names'].split(';')):]), axis=1)

    ncbi_arthropoda = ncbi_arthropoda_.iloc[1:]
    return ncbi_arthropoda, gbif_arthropoda




def match_datasets_2(query_dataset, target_dataset, model, relevant_features, threshold):

    target_dataset_2 = target_dataset[~target_dataset['ncbi_canonicalName'].str.contains(r'\d')]
    
    target = list(set(target_dataset_2.ncbi_target_string))
    query = list(set(query_dataset.gbif_taxonomy))
    
    vectorizer = TfidfVectorizer(analyzer=ngrams, lowercase=True)
    tfidf = vectorizer.fit_transform(target)

    # Il tuo codice originale
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

    def geneate_taxonomic_tree(df_matched_3, df_unmatched_3):
    
    tree = generate_tree(df_matched_3, df_unmatched_3)
    #Save tree
    save_tree(tree, "./taxonomical_tree.txt")
    #Index tree
    indexed_tree = index_tree(tree, include_name=False)
    indexed_tree_data = tree_to_dataframe(indexed_tree)
    df_indexed_tree = pd.DataFrame(indexed_tree_data).tail(-1)
    df_indexed_tree['ncbi_id'] = df_indexed_tree['ncbi_id'].fillna(-1).astype(int)
    df_indexed_tree['gbif_taxon_id'] = df_indexed_tree['gbif_taxon_id'].fillna(-1).astype(int)
    #Save tree
    save_tree(tree, "./indexed_taxonomical_tree.txt")
    doubles_2 = list(set(df_indexed_tree[df_indexed_tree.gbif_taxon_id.duplicated()].gbif_taxon_id))[0:-1]
    df_indexed_tree[df_indexed_tree.gbif_taxon_id.isin(doubles_2)].sort_values(["gbif_taxon_id", "ncbi_id"])
    
    # Copia dataframe 
    df_indexed_tree_ = df_indexed_tree.copy()
    # Effettua il merge basato sulla colonna "ncbi_id" di test con "ncbi_id" di ncbi_dataset
    merged_df1 = pd.merge(df_indexed_tree_, ncbi_dataset[1], left_on=df_indexed_tree_['ncbi_id'].astype(int), right_on='ncbi_id', how='left')
    # Effettua il merge basato sulla colonna "gbif_taxon_id" di test con "lataxonID" di gbif_dataset
    final_dataset = pd.merge(merged_df1, gbif_dataset[1], left_on=merged_df1['gbif_taxon_id'].astype(int), right_on='taxonID', how='left')
    final_dataset_ = final_dataset.copy()
    # Crea la nuova colonna nel dataframe utilizzando il dizionario
    final_dataset_['gbif_synonyms_names'] = final_dataset_['taxonID'].map(lambda x: '; '.join(map(str, synonym_dict_names.get(x, []))).lower())
    final_dataset_['gbif_synonyms_ids'] = final_dataset_['taxonID'].map(lambda x: '; '.join(map(str, synonym_dict_ids.get(x, []))).lower())
    final_dataset_['ncbi_synonyms_names'] = final_dataset_['ncbi_id'].map(lambda x: '; '.join(map(str, ncbi_synonyms.get(x, []))).lower())
    final_dataset_['id'] = range(1, len(final_dataset_) + 1)
    final_dataset_ = final_dataset_.fillna(-1)
    final_dataset_ = final_dataset_[["id", "Index", "ncbi_id", "gbif_taxon_id",  "ncbi_canonicalName", "canonicalName", 'gbif_synonyms_ids', "gbif_synonyms_names", "ncbi_synonyms_names"]]
    final_dataset_.columns = ["id", "path", "ncbi_taxon_id", "gbif_taxon_id", "ncbi_canonical_name", "gbif_canonical_name", "gbif_synonyms_ids", "gbif_synonyms_names", "ncbi_synonyms_names"]
    final_dataset_.to_csv("./taxonomic_tree_v3.0.txt", index=False)
    return(final_dataset)