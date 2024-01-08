
import gensim
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def get_gbif_synonyms(gbif_dataset):
    """
    Extracts synonyms from a GBIF dataset.

    Args:
    gbif_dataset (DataFrame): A pandas DataFrame containing GBIF data.

    Returns:
    tuple: Two dictionaries mapping accepted taxonomic IDs to their synonyms' names and IDs.
    """
    df_gbif_synonyms = gbif_dataset[1][gbif_dataset[1].taxonomicStatus == 'synonym']
    
    # Create the dictionary
    gbif_synonyms_names = {}
    gbif_synonyms_ids = {}
    
    for index, row in df_gbif_synonyms.iterrows():
        accepted_id = row['acceptedNameUsageID']
        canonical_name = row['canonicalName']
        canonical_name_id = row['taxonID']
        
        # Check if canonical_name is not null before adding it to the dictionary
        if not pd.isnull(canonical_name):
            if accepted_id in gbif_synonyms_names:
                # If the accepted_id is already in the dictionary, add the canonical_name
                gbif_synonyms_names[accepted_id].append(canonical_name)
                gbif_synonyms_ids[accepted_id].append(canonical_name_id)
            else:
                # If the accepted_id is not yet in the dictionary, create a new list with the canonical_name
                gbif_synonyms_names[accepted_id] = [canonical_name]
                gbif_synonyms_ids[accepted_id] = [canonical_name_id]
    return gbif_synonyms_names, gbif_synonyms_ids

def get_ncbi_synonyms(names_path):
    """
    Creates a dictionary associating TaxIDs with their names (including synonyms) from NCBI data.

    Args:
    names_path (str): Path to the NCBI names.dmp file.

    Returns:
    dict: Dictionary mapping TaxIDs to a list of names.
    """
    names_dict = {}
    
    # Read names.dmp
    with open(names_path, 'r') as names_file:
        for line in names_file:
            parts = line.strip().split('|')
            taxid = int(parts[0].strip())
            name = parts[1].strip()
            name_type = parts[3].strip()  # Type of name, e.g., synonym, equivalent name, scientific name
    
            if name_type in ['acronym', 'blast name', 'common name', 'equivalent name', 'genbank acronym', 'genbank common name', 'synonym']:
                if taxid not in names_dict:
                    names_dict[taxid] = []
                names_dict[taxid].append(name)
    return names_dict


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