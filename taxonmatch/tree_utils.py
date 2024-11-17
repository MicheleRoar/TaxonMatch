import re
import ete3
import json
import pickle
import pandas as pd
import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import JsonExporter
from taxonmatch.loader import load_gbif_dictionary, load_ncbi_dictionary


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


def reroot_tree(tree, root_name=None):
    """
    Reroots the tree to start from the specified root node.

    Args:
    tree (AnyNode or similar tree node): The current tree.
    root_name (str, optional): The name of the node to use as the new root, case-insensitive.

    Returns:
    AnyNode: The new root of the tree if root_name is specified and found, otherwise the original tree.
    """
    if root_name is not None:
        root_name = root_name.lower()  # Convert root_name to lowercase for case-insensitive search
        new_root = find_node_by_name(tree, root_name) 
        if new_root is None:
            print("Root node not found.")
            return None  # Return None if the root node is not found
        return new_root
    return tree  # Return the original tree if no root_name is specified


def print_tree(tree, root_name=None):
    """
    Prints the structure of a tree sorted alphabetically starting from a specified root node.

    Args:
    tree (AnyNode or a similar tree node): The root node to start from.
    root_name (str, optional): The name of the node to use as the root for printing, case-insensitive.

    Each node of the tree may have attributes 'ncbi_id' and 'gbif_taxon_id'.
    The nodes are printed with their names, and IDs are included if available.
    """
    def sort_children(node):
        # Ensure that node.children is a list of nodes and sort it
        if isinstance(node, list):
            return sorted(node, key=lambda child: child.name.lower())
        else:
            return sorted(node.children, key=lambda child: child.name.lower())

    if root_name is not None:
        root_name = root_name.lower()  # Convert root_name to lowercase for case-insensitive search
        tree = find_node_by_name(tree, root_name)
        if tree is None:
            print("Root node not found.")
            return

    # Print the tree from the new root node, sorted alphabetically
    for pre, fill, node in RenderTree(tree, childiter=sort_children):
        ncbi_id = getattr(node, 'ncbi_id', None)
        gbif_id = getattr(node, 'gbif_taxon_id', None)
        id_info = f" (NCBI ID: {ncbi_id}, GBIF ID: {gbif_id})" if ncbi_id or gbif_id else ""
        print(f"{pre}{node.name}{id_info}")

def tree_to_newick(node):
    """
    Recursively converts an AnyNode tree to Newick format.
    """
    if not node.children:
        return node.name
    children_newick = ",".join([tree_to_newick(child) for child in node.children])
    return f"({children_newick}){node.name}"

def save_tree(tree, path, output_format='txt'):
    """
    Saves a tree structure to a file with identification details for each node in various formats.
    Args:
    tree (AnyNode or similar tree node): The root of the tree to be saved.
    path (str): Path to the file where the tree will be saved.
    output_format (str): Format to save the tree. Supported formats: 'txt', 'newick', 'json'.
    """
    def alphabetical_sort(node):
        return node.name.lower()  # Sort nodes alphabetically by name

    def sort_children(node):
        # Sort the children of the current node, if it has any
        if hasattr(node, 'children') and node.children:
            node.children = sorted(node.children, key=lambda child: child.name.lower())
            # Recursively sort the children of each child node
            for child in node.children:
                sort_children(child)

    # Sort the tree before saving
    sort_children(tree)

    if output_format == 'txt':
        with open(path, 'w') as f:
            # Print the tree with IDs to the file
            for pre, fill, node in RenderTree(tree):
                ncbi_id = getattr(node, 'ncbi_id', None)
                gbif_id = getattr(node, 'gbif_taxon_id', None)
                id_info = f" (NCBI ID: {ncbi_id}, GBIF ID: {gbif_id})" if ncbi_id or gbif_id else ""
                f.write(f"{pre}{node.name}{id_info}\n")
        print(f"The tree is saved as TXT in the file: {path}.")

    elif output_format == 'newick':
        newick_representation = tree_to_newick(tree) + ';'
        with open(path, 'w') as f:
            f.write(newick_representation)
        print(f"The tree is saved as Newick in the file: {path}.")

    elif output_format == 'json':
        exporter = JsonExporter(indent=2, sort_keys=True)
        json_data = exporter.export(tree)
        with open(path, 'w') as f:
            f.write(json_data)
        print(f"The tree is saved as JSON in the file: {path}.")

    else:
        raise ValueError(f"Unsupported format: {output_format}. Supported formats are 'txt', 'newick', and 'json'.")


def index_tree(node, index_list=None, include_name=False):
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

def convert_tree_to_dataframe(tree, query_dataset, target_dataset, path, index=False):
    """
    Converts a taxonomical tree structure into a pandas DataFrame, merges it with external datasets based on NCBI and GBIF identifiers, then enriches it with synonyms from preloaded dictionaries. Finally, it saves the enriched dataset to a specified CSV file path.
    
    The function performs several key operations: indexing the tree, merging with NCBI and GBIF datasets, enriching with synonyms, and exporting to CSV. It assumes the presence of 'target_dataset' and 'query_dataset' dataframes in the scope, as well as preloaded synonym dictionaries for both NCBI and GBIF.
    
    Args:
    tree (object): The taxonomical tree to be converted, structured in a format compatible with 'index_tree' and 'tree_to_dataframe' methods.
    path (str): The file path where the final CSV file will be saved.
    
    Returns:
    DataFrame: The final, enriched dataset containing original tree data, external dataset information, synonyms, and a unique ID for each entry. The DataFrame is also saved to a CSV file at the specified path.
    """
    indexed_tree = tree.copy()
    
    if index:
        # Index the tree
        indexed_tree = index_tree(tree, include_name=False)
    
    indexed_tree_data = tree_to_dataframe(indexed_tree)
    df_indexed_tree = pd.DataFrame(indexed_tree_data).tail(-1)
    
    df_indexed_tree['ncbi_id'] = df_indexed_tree['ncbi_id'].fillna(-1).astype(int)
    df_indexed_tree['gbif_taxon_id'] = df_indexed_tree['gbif_taxon_id'].fillna(-1).astype(int)
    
    
    # Ensure the modifications are done directly on the DataFrame
    target_dataset.loc[:, 'ncbi_id'] = target_dataset['ncbi_id'].astype(int)
    df_indexed_tree.loc[:, 'ncbi_id'] = df_indexed_tree['ncbi_id'].astype(int)
    
    # Perform merge based on "ncbi_id" column of test with "ncbi_id" of ncbi_dataset
    merged_df1 = pd.merge(df_indexed_tree, target_dataset, left_on='ncbi_id', right_on='ncbi_id', how='left')
    
    # Perform merge based on "gbif_taxon_id" column of test with "taxonID" of gbif_dataset
    final_dataset = pd.merge(merged_df1, query_dataset, left_on=merged_df1['gbif_taxon_id'], right_on='taxonID', how='left')
    final_dataset_ = final_dataset.copy()
    
    # Load GBIF and NCBI dictionaries
    gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids = load_gbif_dictionary()
    ncbi_synonyms_names, ncbi_synonyms_ids = load_ncbi_dictionary()
        
    # Create new columns in the dataframe using the dictionaries
    final_dataset_['gbif_synonyms_names'] = final_dataset_['taxonID'].map(lambda x: '; '.join(map(str, gbif_synonyms_ids.get(x, []))))
    final_dataset_['gbif_synonyms_ids'] = final_dataset_['taxonID'].map(lambda x: '; '.join(map(str, gbif_synonyms_ids_to_ids.get(x, []))))
    final_dataset_['ncbi_synonyms_names'] = final_dataset_['ncbi_id'].map(lambda x: '; '.join(map(str, ncbi_synonyms_ids.get(x, []))))
    final_dataset_['id'] = range(1, len(final_dataset_) + 1)  # Assign a unique ID to each row
    
    final_dataset_ = final_dataset_.infer_objects(copy=False).fillna(-1)
    final_dataset_ = final_dataset_.replace([-1, '-1', ""], None)
    
    # Select and rename columns
    final_dataset_ = final_dataset_[["id", "Index", "ncbi_id", "gbif_taxon_id", "ncbi_canonicalName", "canonicalName", 'gbif_synonyms_ids', "gbif_synonyms_names", "ncbi_synonyms_names", "gbif_taxonomy", "ncbi_target_string"]]
    final_dataset_.columns = ["id", "path", "ncbi_taxon_id", "gbif_taxon_id", "ncbi_canonical_name", "gbif_canonical_name", "gbif_synonyms_ids", "gbif_synonyms_names", "ncbi_synonyms_names", "gbif_taxonomy", "ncbi_target_string"]
    
    final_dataset_.to_csv(path, index=False)  # Save the final dataset to a CSV
    return final_dataset_



def find_synonyms(input_term, ncbi_data, gbif_data):
    # Reset accepted_value and synonyms
    accepted_value_ncbi = None
    accepted_value_gbif = None
    ncbi_synonyms = []
    gbif_synonyms = []

    # Step 1: Check if the input is an accepted value or synonym in NCBI
    if input_term in ncbi_data:
        accepted_value_ncbi = input_term
        ncbi_synonyms = ncbi_data[accepted_value_ncbi]
    else:
        for key, synonyms in ncbi_data.items():
            if input_term in synonyms:
                accepted_value_ncbi = key
                ncbi_synonyms = synonyms
                break

    # Step 2: Search in GBIF for the accepted value or its synonyms
    if input_term in gbif_data:
        gbif_synonyms = gbif_data[input_term]
        accepted_value_gbif = input_term
    else:
        for key, synonyms in gbif_data.items():
            if input_term in synonyms or any(ncbi_synonym in synonyms for ncbi_synonym in ncbi_synonyms):
                gbif_synonyms = gbif_data[key]
                accepted_value_gbif = key  # Update accepted value based on GBIF synonym match
                break

    # Step 3: Ensure the accepted value is updated to the corresponding NCBI key, if found in GBIF synonyms
    for key, synonyms in ncbi_data.items():
        if accepted_value_gbif in synonyms:
            accepted_value_ncbi = key  # Update the accepted value to the NCBI key

    # Step 4: Check if any NCBI synonyms correspond to GBIF keys and add their synonyms
    for ncbi_synonym in ncbi_synonyms:
        if ncbi_synonym in gbif_data:
            gbif_synonyms.extend(gbif_data[ncbi_synonym])
            accepted_value_gbif = ncbi_synonym  # If NCBI synonym is a GBIF key, update accepted GBIF name

    # Add synonyms from the NCBI key if it is present in GBIF as well
    if accepted_value_ncbi in gbif_data:
        gbif_synonyms.extend(gbif_data[accepted_value_ncbi])

    # Now add the labels (NCBI / GBIF)
    accepted_value = f"{accepted_value_ncbi} (NCBI)"
    if accepted_value_gbif and accepted_value_gbif != accepted_value_ncbi:
        accepted_value += f" / {accepted_value_gbif} (GBIF)"

    # Ensure the NCBI synonyms are correctly populated
    if accepted_value_ncbi in ncbi_data:
        ncbi_synonyms = ncbi_data[accepted_value_ncbi]

    # Final result
    return {
        'Accepted Value': accepted_value,
        'GBIF Synonyms': list(set(gbif_synonyms)),
        'NCBI Synonyms': list(set(ncbi_synonyms))
    }
    



    