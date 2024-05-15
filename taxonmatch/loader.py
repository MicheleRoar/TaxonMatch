import re
import pandas as pd
import numpy as np
from importlib import resources
import pickle


def load_xgb_model():
    with resources.open_binary('taxonmatch.files.models', 'xgb_model.pkl') as file:
        model = pickle.load(file)
    return model

def load_gbif_dictionary():
    with resources.open_binary('taxonmatch.files.dictionaries', 'gbif_dictionaries.pkl') as file:
        gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids = pickle.load(file)
    return gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids

def load_ncbi_dictionary():
    with resources.open_binary('taxonmatch.files.dictionaries', 'ncbi_dictionaries.pkl') as file:
        ncbi_synonyms_names, ncbi_synonyms_ids = pickle.load(file)
    return ncbi_synonyms_names, ncbi_synonyms_ids