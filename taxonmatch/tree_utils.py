import re
import ete3
import pickle
import pandas as pd
import numpy as np
from anytree import Node, RenderTree


def find_node_by_name(tree, name):
    """
    Recursively find a node by name in the tree, case-insensitively.

    Args:
    tree (AnyNode or similar tree node): The current node to check.
    name (str): The name of the node to find, case-insensitively.

    Returns:
    AnyNode or similar tree node: The node with the specified name, or None if not found.
    """
    if tree.name.lower() == name.lower():
        return tree
    for child in tree.children:
        result = find_node_by_name(child, name)
        if result is not None:
            return result
    return None

def print_tree(tree, root_name=None):
    """
    Prints a tree structure starting from a specified root node, with identification details for each node,
    allowing for case-insensitive node name specification.

    Args:
    tree (AnyNode or similar tree node): The initial root of the tree.
    root_name (str, optional): The name of the node to use as the new root for printing, case-insensitively.

    Each node of the tree is expected to potentially have 'ncbi_id' and 'gbif_taxon_id' attributes.
    Nodes are printed with their names, and IDs are included if available.
    """
    if root_name is not None:
        root_name = root_name.lower()  # Convert root_name to lowercase to ensure case-insensitivity
        tree = find_node_by_name(tree, root_name)
        if tree is None:
            print("Root node not found.")
            return

    # Print the tree from the new root with identification details
    for pre, fill, node in RenderTree(tree):
        ncbi_id = getattr(node, 'ncbi_id', None)
        gbif_id = getattr(node, 'gbif_taxon_id', None)
        id_info = f" (NCBI ID: {ncbi_id}, GBIF ID: {gbif_id})" if ncbi_id or gbif_id else ""
        print(f"{pre}{node.name}{id_info}")

def save_tree(tree, path):
    """
    Saves a tree structure to a file with identification details for each node.

    Args:
    tree (AnyNode or similar tree node): The root of the tree to be saved.
    path (str): Path to the file where the tree will be saved.

    Each node of the tree is expected to potentially have 'ncbi_id' and 'gbif_taxon_id' attributes.
    The function saves the tree to a file, each node with its name and IDs (if available), and prints a confirmation message.
    """
    # Open a file in write mode
    with open(path, 'w') as f:
        # Print the tree with IDs to the file
        for pre, fill, node in RenderTree(tree):
            ncbi_id = getattr(node, 'ncbi_id', None)
            gbif_id = getattr(node, 'gbif_taxon_id', None)
            id_info = f" (NCBI ID: {ncbi_id}, GBIF ID: {gbif_id})" if ncbi_id or gbif_id else ""
            f.write(f"{pre}{node.name}{id_info}\n")

    print(f"The tree is saved in the file: {path}.")



def index_tree(node, index_list=None, include_name=True):
    """
    Recursively indexes a tree, assigning each node a hierarchical index based on its position within the tree.
    
    Args:
    node (AnyNode or similar): The current node to process.
    index_list (list, optional): A list to hold indices as we traverse the tree. Defaults to None, which initializes an empty list.
    include_name (bool): Whether to include the original name of the node in its new indexed name. Defaults to True.

    Returns:
    node: The node with its name updated to include its hierarchical index.

    The function modifies each node's name to include a hierarchical index (e.g., 1.1.1.1 for a fourth-level child) followed by the original node name if include_name is True. This allows for easy identification of each node's position in the tree.
    """
    if index_list is None:
        index_list = []  # Initialize index list if not provided

    # Generate index label for all nodes except the root if include_name is True
    if len(index_list) > 0 or (len(index_list) == 0 and include_name):
        node_name = node.name if include_name else ""  # Include node's name based on the flag
        node.name = '.'.join(str(i) for i in index_list) + (" " + node_name if node_name else "")

    for i, child in enumerate(node.children):
        index_list.append(i + 1)  # Append the current child's index
        index_tree(child, index_list, include_name)  # Recurse for the child
        index_list.pop()  # Remove the current child's index after recursion

    return node


def generate_dataframe(node):
    """
    Generates a DataFrame containing NCBI IDs, GBIF Taxon IDs, and the hierarchical index of each node in a tree.

    Args:
    node (AnyNode or similar): The root of the tree from which to generate the DataFrame.

    Returns:
    DataFrame: A DataFrame with columns 'ncbi_id', 'gbif_taxon_id', and 'Index', representing each node's NCBI ID, GBIF Taxon ID, and hierarchical index within the tree, respectively.

    This function traverses the tree starting from the given node and collects the NCBI ID, GBIF Taxon ID, and the hierarchical index (node.name) for each node, then stores this information in a DataFrame.
    """
    data = []
    # Traverse the tree and collect data
    for pre, fill, node in RenderTree(node):
        ncbi_id = getattr(node, 'ncbi_id', '')  # Get NCBI ID if available, else default to an empty string
        gbif_id = getattr(node, 'gbif_taxon_id', '')  # Get GBIF Taxon ID if available, else default to an empty string
        data.append((ncbi_id, gbif_id, node.name))  # Collect all the necessary details

    # Create a DataFrame with specified columns
    df = pd.DataFrame(data, columns=['ncbi_id', 'gbif_taxon_id', 'Index'])
    return df


import pandas as pd

def export_tree(tree, ncbi_dataset, gbif_dataset, path):
    """
    Exports a tree structure into a CSV file with detailed taxonomic information by merging it with NCBI and GBIF datasets.

    Args:
    tree (AnyNode or similar): The root of the tree structure to be exported.
    ncbi_dataset (DataFrame): The dataset containing NCBI taxonomic data.
    gbif_dataset (DataFrame): The dataset containing GBIF taxonomic data.
    path (str): File path where the CSV will be saved.

    This function generates a DataFrame from a tree, processes and converts taxonomic IDs, merges additional taxonomic information from NCBI and GBIF datasets,
    extracts and incorporates synonyms from GBIF, and finally exports the complete data to a CSV file.
    """
    # Generate DataFrame from tree
    df_final = generate_dataframe(tree)

    # Remove the first empty row
    df_final = df_final.tail(-1)

    # Convert columns to int, treating empty values as -1
    df_final['ncbi_id'] = pd.to_numeric(df_final['ncbi_id'], errors='coerce').fillna(-1).astype('int64')
    df_final['gbif_taxon_id'] = pd.to_numeric(df_final['gbif_taxon_id'], errors='coerce').fillna(-1).astype('int64')

    # Merge with NCBI dataset on 'ncbi_id'
    merged_df1 = pd.merge(df_final, ncbi_dataset, left_on='ncbi_id', right_on='ncbi_id', how='left')

    # Merge with GBIF dataset on 'gbif_taxon_id'
    final_dataset = pd.merge(merged_df1, gbif_dataset, left_on='gbif_taxon_id', right_on='taxonID', how='left')
    
    # Extract synonyms from GBIF
    synonym_dict_names, synonym_dict_ids = get_gbif_synonyms(gbif_dataset)
    
    # Create new columns in the dataframe using the dictionary
    final_dataset['synonyms_names'] = final_dataset['taxonID'].map(lambda x: '; '.join(map(str, synonym_dict_names.get(x, []))).lower())
    final_dataset['synonyms_ids'] = final_dataset['taxonID'].map(lambda x: '; '.join(map(str, synonym_dict_ids.get(x, []))).lower())
    
    # Select relevant columns for output
    final_dataset = final_dataset[["ncbi_id", "gbif_taxon_id", "Index", "ncbi_canonicalName", "canonicalName", "synonyms_names", 'synonyms_ids']]
    
    # Save DataFrame to a CSV file
    final_dataset.to_csv(path, index=False)

    print(f"Tree data has been exported to {path}.")


def tree_to_dataframe(tree):
    """
    Converts a tree structure into a list of dictionaries, each representing a node within the tree with its index and taxonomic IDs.

    Args:
    tree (Tree): The tree structure to be converted. The tree is assumed to be of a type that has a .traverse() method (like in Bio.Phylo).

    Returns:
    list: A list of dictionaries, where each dictionary contains the index, NCBI ID, and GBIF Taxon ID of a node.

    This function traverses each node of the given tree and collects the node's name and available taxonomic IDs into a list of dictionaries. Each node's data includes an index (node's name), an NCBI ID, and a GBIF Taxon ID if available in the node's features.
    """
    data = []
    # Traverse the tree and collect data from each node
    for node in tree.traverse():
        node_data = {
            "Index": node.name,  # Node's name used as an index
            "ncbi_id": node.ncbi_id if "ncbi_id" in node.features else None,  # Collect NCBI ID if available
            "gbif_taxon_id": node.gbif_taxon_id if "gbif_taxon_id" in node.features else None,  # Collect GBIF Taxon ID if available
        }
        data.append(node_data)
    return data


def manage_duplicated_branches(matched_df, unmatched_df):
    """
    Manages duplicated branches in matched and unmatched dataframes by standardizing names based on a mapping derived from matched records.

    Args:
    matched_df (DataFrame): A dataframe with matched entries that may have discrepancies in canonical naming.
    unmatched_df (DataFrame): A dataframe with unmatched entries that need name standardization.

    Returns:
    tuple: A tuple containing the original matched dataframe and the corrected unmatched dataframe.

    This function identifies common entries with different canonical names between 'canonicalName' and 'ncbi_canonicalName' columns in the matched dataframe.
    It then creates a mapping from these names and applies corrections to the 'gbif_taxonomy' column in the unmatched dataframe using this mapping.
    """
    # Find entries with different canonical names
    common = matched_df.query("canonicalName != ncbi_canonicalName")
    pattern = r'^\b\w+\b$'  # Matches a whole word
    filtered_df = common[common['canonicalName'].str.match(pattern, na=False)]
    gbif_to_ncbi_map = dict(zip(filtered_df['canonicalName'].str.lower(), filtered_df['ncbi_canonicalName'].str.lower()))

    # Copy unmatched dataframe for correction
    corrected_unmatched_df = unmatched_df.copy()

    # Convert 'gbif_taxonomy' column to string
    corrected_unmatched_df['gbif_taxonomy'] = corrected_unmatched_df['gbif_taxonomy'].astype(str)
    
    # Function to replace substrings using the created mapping
    def replace_with_dict(text):
        for key, value in gbif_to_ncbi_map.items():
            text = re.sub(r'\b{}\b'.format(re.escape(key)), value, text)
        return text
    
    # Apply the replacement to the 'gbif_taxonomy' column
    corrected_unmatched_df['gbif_taxonomy_corrected'] = corrected_unmatched_df['gbif_taxonomy'].apply(replace_with_dict)
    
    return matched_df, corrected_unmatched_df



def generate_taxonomic_tree(df_matched, df_unmatched):
    """
    Generates a phylogenetic tree from matched and unmatched DataFrame inputs, integrating taxonomic IDs and names.

    Args:
    df_matched (DataFrame): Contains matched taxonomic data including NCBI lineage names and IDs.
    df_unmatched (DataFrame): Contains unmatched GBIF taxonomic data needing corrections.

    Returns:
    tree (ete3.Tree): A tree object populated with taxonomic nodes and IDs from both NCBI and GBIF sources.

    This function processes matched and unmatched dataframes to create a comprehensive taxonomic tree. 
    It handles discrepancies in naming through a management function and constructs a tree using names and IDs.
    """
    
    # Process dataframes to manage duplicated branches and resolve naming discrepancies
    df_matched_processed, df_unmatched_processed = manage_duplicated_branches(df_matched, df_unmatched)

    # Initialize the phylogenetic tree
    tree = ete3.Tree()

    # Dictionaries to track nodes by taxon name and identifiers by taxon ID
    node_dict = {}
    taxon_id_dict = {}

    # Insert information from processed matched dataframe
    for row in df_matched_processed.itertuples(index=False):
        ncbi_lineage_names = row.ncbi_lineage_names.lower().split(';')
        ncbi_lineage_ids = row.ncbi_lineage_ids.split(';')
        ncbi_taxon_id = row.taxonID

        parent_node = tree
        for i, taxon in enumerate(ncbi_lineage_names):
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

            # Add GBIF ID as the taxonID for leaves
            if i == len(ncbi_lineage_names) - 1:
                parent_node.add_feature('gbif_taxon_id', ncbi_taxon_id)

    # Insert information from processed unmatched dataframe
    for row in df_unmatched_processed.itertuples(index=False):
        gbif_taxonomy = row.gbif_taxonomy_corrected.split(';')
        gbif_taxonomy_ids = row.gbif_taxonomy_ids.split(';')

        # Ensure the number of elements is the same
        if len(gbif_taxonomy) != len(gbif_taxonomy_ids):
            continue

        parent_node = tree
        for i, taxon in enumerate(gbif_taxonomy):
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

def convert_tree_to_dataframe(tree, ncbi_arthropoda, gbif_arthropoda, path, index=False):
    """
    Converts a taxonomical tree structure into a pandas DataFrame, merges it with external datasets based on NCBI and GBIF identifiers, then enriches it with synonyms from preloaded dictionaries. Finally, it saves the enriched dataset to a specified CSV file path.

    The function performs several key operations: indexing the tree, merging with NCBI and GBIF datasets, enriching with synonyms, and exporting to CSV. It assumes the presence of 'ncbi_arthropoda' and 'gbif_arthropoda' dataframes in the scope, as well as preloaded synonym dictionaries for both NCBI and GBIF.

    Args:
    tree (object): The taxonomical tree to be converted, structured in a format compatible with 'index_tree' and 'txm.tree_to_dataframe' methods.
    path (str): The file path where the final CSV file will be saved.

    Returns:
    DataFrame: The final, enriched dataset containing original tree data, external dataset information, synonyms, and a unique ID for each entry. The DataFrame is also saved to a CSV file at the specified path.
    """
    indexed_tree = tree.copy()

    if index:
        # Index the tree
        indexed_tree = index_tree(tree, include_name=False)

    indexed_tree_data = tree_to_dataframe(indexed_tree)
    df_indexed_tree = pd.DataFrame(indexed_tree_data).tail(-1)  # Remove the first row if it's a header or unwanted
    df_indexed_tree['ncbi_id'] = df_indexed_tree['ncbi_id'].fillna(-1).astype(int)
    df_indexed_tree['gbif_taxon_id'] = df_indexed_tree['gbif_taxon_id'].fillna(-1).astype(int)

    # Save the tree
    # save_tree(tree, "./indexed_taxonomical_tree.txt")
    # Find duplicates in 'gbif_taxon_id' column 
    # doubles_2 = list(set(df_indexed_tree[df_indexed_tree.gbif_taxon_id.duplicated()].gbif_taxon_id))[0:-1]
    # Sort rows with duplicated 'gbif_taxon_id' by 'gbif_taxon_id' and 'ncbi_id'
    # df_indexed_tree[df_indexed_tree.gbif_taxon_id.isin(doubles_2)].sort_values(["gbif_taxon_id", "ncbi_id"])
    
    # Copy the dataframe
    df_indexed_tree_ = df_indexed_tree.copy()
    
    # Perform merge based on "ncbi_id" column of test with "ncbi_id" of ncbi_dataset
    merged_df1 = pd.merge(df_indexed_tree_, ncbi_arthropoda, left_on=df_indexed_tree_['ncbi_id'].astype(int), right_on='ncbi_id', how='left')
    
    # Perform merge based on "gbif_taxon_id" column of test with "taxonID" of gbif_dataset
    final_dataset = pd.merge(merged_df1, gbif_arthropoda, left_on=merged_df1['gbif_taxon_id'].astype(int), right_on='taxonID', how='left')
    final_dataset_ = final_dataset.copy()
    
    # Load GBIF and NCBI dictionaries
    with open('./files/dictionaries/gbif_dictionaries.pkl', 'rb') as f:
        gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids = pickle.load(f)
    
    with open('./files/dictionaries/ncbi_dictionaries.pkl', 'rb') as f:
        ncbi_synonyms_names, ncbi_synonyms_ids = pickle.load(f)
        
    # Create new columns in the dataframe using the dictionaries
    final_dataset_['gbif_synonyms_names'] = final_dataset_['taxonID'].map(lambda x: '; '.join(map(str, gbif_synonyms_ids.get(x, []))))
    final_dataset_['gbif_synonyms_ids'] = final_dataset_['taxonID'].map(lambda x: '; '.join(map(str, gbif_synonyms_ids_to_ids.get(x, []))))
    final_dataset_['ncbi_synonyms_names'] = final_dataset_['ncbi_id'].map(lambda x: '; '.join(map(str, ncbi_synonyms_ids.get(x, []))))
    final_dataset_['id'] = range(1, len(final_dataset_) + 1)  # Assign a unique ID to each row
    final_dataset_ = final_dataset_.fillna(-1)  # Fill missing values with -1
    # Select and rename columns
    final_dataset_ = final_dataset_[["id", "Index", "ncbi_id", "gbif_taxon_id", "ncbi_canonicalName", "canonicalName", 'gbif_synonyms_ids', "gbif_synonyms_names", "ncbi_synonyms_names"]]
    final_dataset_.columns = ["id", "path", "ncbi_taxon_id", "gbif_taxon_id", "ncbi_canonical_name", "gbif_canonical_name", "gbif_synonyms_ids", "gbif_synonyms_names", "ncbi_synonyms_names"]
    final_dataset_.fillna(-1, inplace=True)

    # Standardize missing values
    final_dataset_.replace({-1: np.nan, 'nan': np.nan, '': np.nan, 'None': np.nan, '-1.0': np.nan}, inplace=True)

    # Ensure that all columns use appropriate data types
    for col in final_dataset_.columns:
        if pd.api.types.is_numeric_dtype(final_dataset_[col]):
            # Maintain float for a uniform representation of NaN
            final_dataset_[col] = pd.to_numeric(final_dataset_[col], errors='coerce').astype(float)
        else:
            final_dataset_[col] = final_dataset_[col].astype('object')


    final_dataset_.to_csv(path, index=False)  # Save the final dataset to a CSV
    return(final_dataset_)