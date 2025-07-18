import re
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from importlib import resources
import pickle


def load_xgb_model():
    directory = resources.files('taxonmatch.files.models')
    file_path = directory / 'xgb_model.json'
    model = XGBClassifier()
    model.load_model(str(file_path))
    return model

def load_gbif_dictionary():
    with resources.open_binary('taxonmatch.files.dictionaries', 'gbif_dictionaries.pkl') as file:
        gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids = pickle.load(file)
    return gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids

def load_ncbi_dictionary():
    with resources.open_binary('taxonmatch.files.dictionaries', 'ncbi_dictionaries.pkl') as file:
        ncbi_synonyms_names, ncbi_synonyms_ids = pickle.load(file)
    return ncbi_synonyms_names, ncbi_synonyms_ids

def load_training_set():
    file_path = resources.files('taxonmatch.files.training_set') / 'training_set.txt'
    with file_path.open('r', encoding='utf-8') as f:
        df = pd.read_csv(f)
    return df

def save_gbif_dictionary(gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids):
    directory = resources.files('taxonmatch.files.dictionaries')
    file_path = directory / 'gbif_dictionaries.pkl'
    
    with open(file_path, 'wb') as f:
        pickle.dump((gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids), f)

def save_ncbi_dictionary(ncbi_synonyms_names, ncbi_synonyms_ids):
    directory = resources.files('taxonmatch.files.dictionaries')
    file_path = directory / 'ncbi_dictionaries.pkl'

    with open(file_path, 'wb') as f:
        pickle.dump((ncbi_synonyms_names, ncbi_synonyms_ids), f)