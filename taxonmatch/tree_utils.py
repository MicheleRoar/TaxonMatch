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
    
    tree_ = tree[0]

    if root_name is not None:
        root_name = root_name.lower()  # Convert root_name to lowercase for case-insensitive search
        new_root = find_node_by_name(tree_, root_name) 
        if new_root is None:
            print("Root node not found.")
            return None  # Return None if the root node is not found
        return new_root, tree[1], tree[2]
    return tree  # Return the original tree if no root_name is specified


def print_tree(tree, root_name=None):
    """
    Prints the structure of a tree sorted alphabetically starting from a specified root node.

    Args:
    tree (AnyNode or similar): The root node to start from.
    root_name (str, optional): The name of the node to use as the root for printing, case-insensitive.

    Each node of the tree may have attributes 'ncbi_id', 'gbif_taxon_id', and 'inat_taxon_id'.
    The nodes are printed with their names, and IDs are included if available.
    """
    def sort_children(node):
        # Check if node.children is a list or attribute and sort accordingly
        if hasattr(node, 'children'):
            return sorted(node.children, key=lambda child: child.name.lower())
        else:
            return node  # Assume node is already a sorted list

    tree = tree[0]
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
        inat_id = getattr(node, 'inat_taxon_id', None)
        
        # Include IDs in the output
        id_info = []
        if ncbi_id: id_info.append(f"NCBI ID: {ncbi_id}")
        if gbif_id: id_info.append(f"GBIF ID: {gbif_id}")
        if inat_id: id_info.append(f"iNaturalist ID: {inat_id}")
        
        id_str = f" ({', '.join(id_info)})" if id_info else ""
        print(f"{pre}{node.name}{id_str}")


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
    def sort_children(node):
        # Sort the children of the current node, if it has any
        if hasattr(node, 'children') and node.children:
            node.children = sorted(node.children, key=lambda child: child.name.lower())
            # Recursively sort the children of each child node
            for child in node.children:
                sort_children(child)

    tree = tree[0]
    # Sort the tree before saving
    sort_children(tree)

    if output_format == 'txt':
        with open(path, 'w') as f:
            # Print the tree with IDs to the file
            for pre, fill, node in RenderTree(tree):
                ncbi_id = getattr(node, 'ncbi_id', None)
                gbif_id = getattr(node, 'gbif_taxon_id', None)
                inat_id = getattr(node, 'inat_taxon_id', None)
                
                # Include IDs in the output
                id_info = []
                if ncbi_id: id_info.append(f"NCBI ID: {ncbi_id}")
                if gbif_id: id_info.append(f"GBIF ID: {gbif_id}")
                if inat_id: id_info.append(f"iNaturalist ID: {inat_id}")
                
                id_str = f" ({', '.join(id_info)})" if id_info else ""
                f.write(f"{pre}{node.name}{id_str}\n")
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
    node: The node with a new attribute 'hierarchical_index' added.

    This function modifies the tree nodes by adding a hierarchical index as a separate attribute.
    """
    if index_list is None:
        index_list = []  # Initialize index list if not provided

    # Generate the hierarchical index as a string
    hierarchical_index = '.'.join(str(i) for i in index_list)

    # Add the hierarchical index as a separate attribute
    node.add_feature('hierarchical_index', hierarchical_index)

    # Optionally modify the name to include the index
    if include_name:
        node.name = f"{hierarchical_index} {node.name}"

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


def tree_to_dataframe(tree, inat_dataset=None):
    """
    Converts a tree structure into a list of dictionaries, each representing a node within the tree 
    with its index and taxonomic IDs, including NCBI, GBIF, and iNaturalist information.

    Args:
    tree (Tree): The tree structure to be converted. The tree is assumed to be of a type that has a 
                 .traverse() method (like in Bio.Phylo or ete3).
    inat_dataset (DataFrame, optional): A dataset containing iNaturalist taxon information. Defaults to None.

    Returns:
    list: A list of dictionaries, where each dictionary contains the index, NCBI ID, GBIF Taxon ID, 
          and optionally iNaturalist ID.
    """
    data = []

    # Traverse the tree and collect data from each node
    for node in tree.traverse():
        node_data = {
            "Index": node.name,  # Node's name used as an index
            "ncbi_id": getattr(node, "ncbi_id", None),  # Collect NCBI ID if available
            "gbif_taxon_id": getattr(node, "gbif_taxon_id", None),  # Collect GBIF Taxon ID if available
        }
        
        # Add iNaturalist ID if the dataset is provided
        if inat_dataset is not None:
            node_data["inat_taxon_id"] = getattr(node, "inat_taxon_id", None)  # iNaturalist Taxon ID
        
        data.append(node_data)

    return data

def clean_synonyms(synonyms):
    if pd.isna(synonyms):
        return None
    unique_synonyms = list(set(s.capitalize() for s in synonyms.split('; ')))
    return '; '.join(unique_synonyms)


def manage_duplicated_branches(matched_df, unmatched_df):
    """
    Manages duplicated branches in matched DataFrame by standardizing NCBI names to GBIF names.

    Args:
    matched_df (DataFrame): A dataframe with matched entries that may have discrepancies in canonical naming.
    unmatched_df (DataFrame): A dataframe with unmatched GBIF entries (not modified).

    Returns:
    tuple: A tuple containing the corrected matched dataframe and the unchanged unmatched dataframe.

    This function identifies common entries where 'canonicalName' (GBIF) differs from 'ncbi_canonicalName' (NCBI).
    It creates a mapping from NCBI names to GBIF names and applies corrections to the 'ncbi_target_string' and 
    'ncbi_lineage_names' columns in matched_df.
    """

    #TO DO: Consider substituition of genus only/ first part subspecies 
    
    # Find entries with different canonical names
    common = matched_df.loc[matched_df['canonicalName'].notna() & (matched_df['canonicalName'] != matched_df['ncbi_canonicalName'])]
    
    if common.empty:
        return matched_df, unmatched_df

    # Create a mapping from NCBI canonical names to GBIF canonical names (all lowercase)
    replace_dict = {k.lower(): v.lower() for k, v in zip(common['ncbi_canonicalName'], common['canonicalName'])}

    pattern = re.compile(r'(^|;)(?:' + '|'.join(map(re.escape, replace_dict.keys())) + r')(?=;|$)', re.IGNORECASE)
    
    # Function to replace only complete taxa names between semicolons
    def fast_replace(text):
        if not isinstance(text, str):  # Evita errori su NaN
            return text
        return pattern.sub(lambda m: m.group(1) + replace_dict[m.group(0).lower().lstrip(";")], text)
    
    # Apply the replacement to 'ncbi_target_string' and 'ncbi_lineage_names'
    matched_df2 = matched_df.copy()
    matched_df2['ncbi_target_string'] = matched_df2['ncbi_target_string'].apply(fast_replace)
    matched_df2['ncbi_lineage_names'] = matched_df2['ncbi_lineage_names'].apply(fast_replace)
    
    return matched_df2, unmatched_df



# Funzione completa aggiornata
def update_subspecies_based_on_synonyms(df_matched, df_unmatched):
    """
    Updates the taxonomy of subspecies based on synonym resolution for their parent species.
    This function filters for `taxonRank == "subspecies"` to optimize performance,
    resolves synonyms for parent species, updates the subspecies taxonomy, 
    and applies the changes to specific columns in df_matched.
    
    Args:
    df_matched (DataFrame): DataFrame with matched taxonomic information.
    df_unmatched (DataFrame): DataFrame with unmatched taxonomic information.
    
    Returns:
    Tuple[DataFrame, DataFrame, dict]: Updated matched and unmatched DataFrames, and the dictionary of changes.
    """

    gbif_synonyms_names, _, _ = load_gbif_dictionary()
    excluded_log = []  # Logs for excluded rows
    modified_log = []  # Logs for modified rows
    taxonomies_to_exclude = []  # List to track taxonomies to exclude
    changes_dict = {}  # Dictionary to store species original -> modified mappings

    def resolve_subspecies(taxonomy, synonyms_dict):
        """Resolve the taxonomy of a subspecies based on its species synonym."""
        if pd.isna(taxonomy):
            return taxonomy
        terms = taxonomy.split(';')
        if len(terms) < 2:
            return taxonomy
        species = terms[-2]  # Penultimate term is the species
        subspecies = terms[-1]  # Last term is the subspecies
        accepted_species = None

        # Identify the accepted name for the species
        for accepted, synonyms in synonyms_dict.items():
            if species.capitalize() in synonyms:
                accepted_species = accepted
                break

        if accepted_species and len(accepted_species.split()) == 3:
            excluded_log.append(
                f"Excluded: {taxonomy} -> Accepted species '{accepted_species}' is a subspecies."
            )
            taxonomies_to_exclude.append(taxonomy)
            return None

        if not accepted_species:
            return taxonomy  # If no synonym found, return the original taxonomy

        # Rule for handling repetition
        subspecies_parts = subspecies.split(" ")
        if len(subspecies_parts) > 1 and subspecies_parts[-1] == subspecies_parts[-2]:
            resolved_subspecies = f"{accepted_species.lower()} {accepted_species.split(' ')[-1]}"
        else:
            resolved_subspecies = f"{accepted_species.lower()} {subspecies_parts[-1]}"

        # Save the species-level change in the dictionary
        changes_dict[species.lower()] = accepted_species.lower()

        # Rebuild the taxonomy with the modified species and subspecies
        terms[-2] = accepted_species.lower()
        terms[-1] = resolved_subspecies.lower()
        return ';'.join(terms)

    # Filter for subspecies
    unmatched_subspecies = df_unmatched[df_unmatched['taxonRank'] == "subspecies"].copy()

    # Resolve taxonomy for filtered rows
    unmatched_subspecies['resolved_gbif_taxonomy'] = unmatched_subspecies['gbif_taxonomy'].apply(
        lambda x: resolve_subspecies(x, gbif_synonyms_names)
    )

    # Log modified rows
    for idx, row in unmatched_subspecies.iterrows():
        original_taxonomy = row['gbif_taxonomy']
        resolved_taxonomy = row['resolved_gbif_taxonomy']
        if pd.isna(resolved_taxonomy):
            continue
        if original_taxonomy != resolved_taxonomy:
            modified_log.append(
                f"Unmatched dataset: {original_taxonomy} -> {resolved_taxonomy}"
            )

    # Remove rows where resolved_gbif_taxonomy is None
    unmatched_subspecies = unmatched_subspecies[unmatched_subspecies['resolved_gbif_taxonomy'].notna()]

    # Update the original column gbif_taxonomy in df_unmatched
    df_unmatched.loc[unmatched_subspecies.index, 'gbif_taxonomy'] = unmatched_subspecies['resolved_gbif_taxonomy']

    # Removes rows from df_unmatched if the converted version already exists in df_matched
    existing_taxa = set(df_matched["gbif_taxonomy"].str.lower())
    df_unmatched = df_unmatched[~df_unmatched["gbif_taxonomy"].str.lower().isin(existing_taxa)]


    # Remove excluded taxonomies from df_unmatched
    df_unmatched = df_unmatched[~df_unmatched['gbif_taxonomy'].isin(taxonomies_to_exclude)]

    # Update specific columns in df_matched using the changes dictionary
    def replace_species_in_column(column, changes_dict):
        """Replace species names in a column using the changes dictionary."""
        return column.apply(
            lambda x: replace_species(x, changes_dict) if pd.notna(x) else x
        )

    def replace_species(text, changes_dict):
        "Replace species names in a string using the changes dictionary and convert to lowercase."
        for original, modified in changes_dict.items():
            text = text.replace(original, modified)
        return text.lower()  # Convert the entire text to lowercase

    # Update columns in df_matched
    for column in ['gbif_taxonomy', 'ncbi_target_string', 'ncbi_lineage_names']:
        if column in df_matched.columns:
            df_matched[column] = replace_species_in_column(df_matched[column], changes_dict)

    # Save the disambiguation log with separate sections
    with open("disambiguation_log.txt", "w") as log_file:
        # Write excluded rows
        log_file.write("Excluded Rows:\n")
        log_file.write("\n".join(excluded_log))
        log_file.write("\n\nModified Rows:\n")
        # Write modified rows
        log_file.write("\n".join(modified_log))

    return df_matched, df_unmatched



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

    #Fix issues related to subspecies name inconsistency
    df_matched_fixed, df_unmatched_fixed, substitution_dict = fix_inconsistent_subspecies(df_matched_processed, df_unmatched_processed)

    # Initialize the phylogenetic tree
    tree = ete3.Tree()

    # Dictionaries to track nodes by taxon name and identifiers by taxon ID
    node_dict = {}
    taxon_id_dict = {}

    # Insert information from processed matched dataframe
    for row in df_matched_fixed.itertuples(index=False):
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
    for row in df_unmatched_fixed.itertuples(index=False):
        gbif_taxonomy = row.gbif_taxonomy.split(';')
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

    return tree, node_dict, taxon_id_dict



def add_inat_taxonomy(tree, df_inat):
    """
    Adds nodes to the existing tree based on the taxonomy in inat_taxonomy.

    Args:
        tree (tuple): A tuple containing:
            tree[0]: The tree structure (ete3.Tree)
            tree[1]: Dictionary of nodes by name (node_dict)
            tree[2]: Dictionary of nodes by taxonomic ID (taxon_id_dict)
        df_inat (DataFrame): A DataFrame with a column 'inat_taxonomy' containing iNaturalist taxonomy.
    
    Returns:
    tuple: Updates tree directly with new nodes and dictionaries.
    """
    # Extract elements from tuples
    phylo_tree, node_dict, taxon_id_dict = tree

    # Iterate over each row of the DataFrame.
    for row in df_inat.itertuples(index=False):
        inat_taxonomy = row.inat_taxonomy.split(';')  # Splitta la tassonomia
        inat_taxon_id = row.inat_taxon_id  # ID del taxon inat

        # Start from the root node.
        parent_node = phylo_tree
        for taxon in inat_taxonomy:
            # Check if the node already exists.
            if taxon not in node_dict:
                # Create a new node and add it to the tree
                new_node = parent_node.add_child(name=taxon)
                node_dict[taxon] = new_node  # Aggiornare il dizionario dei nodi
                parent_node = new_node  # Aggiornare il nodo genitore
            else:
                # Move to the existing node
                parent_node = node_dict[taxon]

        # At the end of the taxonomy, assign the taxon ID to the leaf node
        parent_node.add_feature('inat_taxon_id', inat_taxon_id)
        taxon_id_dict[inat_taxon_id] = parent_node  # Aggiornare il dizionario ID-taxon

    # Return the updated tuple
    return phylo_tree, node_dict, taxon_id_dict



def correct_inconsistent_subspecies(df):

    # Create a dictionary {taxonID -> species_name} for quick reference
    species_dict = df[df["taxonRank"] == "species"].set_index("taxonID")["canonicalName"].to_dict()

    # Filter only subspecies, varieties, and forms
    mask = df["taxonRank"].isin(["subspecies", "variety", "form"])
    df_filtered = df[mask].copy()

    # Retrieve the parent species name using parentNameUsageID.
    df_filtered["species_parent"] = df["parentNameUsageID"].map(species_dict)

    # Check the consistency between the parent species name and the subspecies
    df_filtered["subspecies_first_part"] = df_filtered["canonicalName"].apply(lambda x: " ".join(x.split()[:2]))

    # Identify cases where the first part of the subspecies name does not match the parent species name
    inconsistent_rows = df_filtered.dropna(subset=["subspecies_first_part", "species_parent"])[lambda df: df["subspecies_first_part"] != df["species_parent"]]

    # Replace values in the gbif_taxonomy column based on the specific rule
    def replace_species_name_vectorized(df):
        """
        Replaces species names in the DataFrame efficiently, ensuring that each name is replaced only once per row.
        """

        df = df.copy()
        
        # Convert everything to lowercase to ensure consistency
        df["gbif_taxonomy"] = df["gbif_taxonomy"].astype(str).str.lower()
        df["canonicalName"] = df["canonicalName"].astype(str).str.lower()
        df["subspecies_first_part"] = df["subspecies_first_part"].astype(str).str.lower()
        df["species_parent"] = df["species_parent"].astype(str).str.lower()
        
        # Define the new replaced names
        def compute_replacement(row):
            species_chosen = row["species_parent"]
            subspecies_parts = row["canonicalName"].split()
            
            if len(subspecies_parts) > 1 and subspecies_parts[-1] == subspecies_parts[-2]:
                return f"{species_chosen} {species_chosen.split(' ')[-1]}"
            else:
                return f"{species_chosen} {subspecies_parts[-1]}"
        
        df["resolved_subspecies"] = df.apply(compute_replacement, axis=1)

        # Apply the replacement row by row
        def update_taxonomy(row):
            taxonomy = row["gbif_taxonomy"]
            
            # If canonicalName is present, replace it with resolved_subspecies
            if row["canonicalName"] in taxonomy:
                taxonomy = taxonomy.replace(row["canonicalName"], row["resolved_subspecies"], 1)

            # If subspecies_first_part is present, replace it with species_parent
            if row["subspecies_first_part"] in taxonomy:
                taxonomy = taxonomy.replace(row["subspecies_first_part"], row["species_parent"], 1)
            
            return taxonomy
        
        # Apply the replacement on each row
        df["gbif_taxonomy"] = df.apply(update_taxonomy, axis=1)

        # Remove the temporary column
        df.drop(columns=["resolved_subspecies"], inplace=True)

        return df

    df_filtered_2 = replace_species_name_vectorized(inconsistent_rows)

    # Update the original DataFrame with the corrections
    df.update(df_filtered_2)
    df = df.infer_objects(copy=False) 

    return df


def fix_inconsistent_subspecies(df_matched, df_unmatched):
    """
    Updates the taxonomy of subspecies using NumPy for better performance.
    
    Args:
        df_matched (DataFrame): Matched taxonomy DataFrame.
        df_unmatched (DataFrame): Unmatched taxonomy DataFrame.
    
    Returns:
        Tuple[DataFrame, DataFrame, dict]: Updated DataFrames and substitution dictionary.
    """
    # Extract subspecies rows
    unmatched_array = df_unmatched[df_unmatched['taxonRank'].isin(["subspecies", "form", "variety"])].to_numpy()
    
    if unmatched_array.size == 0:
        return df_matched, df_unmatched, {}

    columns = df_unmatched.columns
    gbif_taxonomy_idx = list(columns).index("gbif_taxonomy")
    gbif_taxonomy_ids_idx = list(columns).index("gbif_taxonomy_ids")

    # Extract relevant columns as NumPy arrays
    gbif_taxonomy = unmatched_array[:, gbif_taxonomy_idx]
    gbif_taxonomy_ids = unmatched_array[:, gbif_taxonomy_ids_idx]

    # Precompute penultimate elements and IDs
    penultimate_element = np.array(
        [x.split(";")[-2] if isinstance(x, str) and len(x.split(";")) > 1 else None for x in gbif_taxonomy]
    )
    penultimate_id = np.array(
        [x.split(";")[-2] if isinstance(x, str) and len(x.split(";")) > 1 else None for x in gbif_taxonomy_ids]
    )

    # Convert penultimate_id in lowercase string for comparison
    penultimate_id = np.char.lower(penultimate_id.astype(str))

    # Create a dictionary {penultimate_id -> canonicalName} based on df_matched
    matched_taxon_ids = df_matched['taxonID'].astype(str).str.lower().to_numpy()
    matched_canonical_names = df_matched['canonicalName'].astype(str).to_numpy()
    taxon_id_to_canonical = dict(zip(matched_taxon_ids, matched_canonical_names))

    # Find the canonicalName for each penultimate_id
    resolved_species = np.array([taxon_id_to_canonical.get(tid, None) for tid in penultimate_id])

    # Filter only the IDs for which a canonicalName was found
    valid_mask = resolved_species != None
    gbif_taxonomy_valid = gbif_taxonomy[valid_mask]
    resolved_species_valid = resolved_species[valid_mask]

    # Create a replacement dictionary {new_subspecies: old_subspecies}
    substitution_dict = {}
    
    # Update the taxonomy of subspecies
    for taxonomy, species_chosen in zip(gbif_taxonomy_valid, resolved_species_valid):
        if not isinstance(taxonomy, str) or len(taxonomy.split(";")) < 2:
            continue  # Ignora valori non validi
    
        terms = taxonomy.split(";")
        subspecies_parts = terms[-1].split(" ")
        old_subspecies = terms[-1] 
    
        if len(subspecies_parts) > 1 and subspecies_parts[-1] == subspecies_parts[-2]:
            resolved_subspecies = f"{species_chosen} {species_chosen.split(' ')[-1]}"
        else:
            resolved_subspecies = f"{species_chosen} {subspecies_parts[-1]}"
    
        old = " ".join(old_subspecies.split(" ")[:2]).lower() + ";" + old_subspecies.lower()
        new = " ".join(resolved_subspecies.split(" ")[:2]).lower() + ";" + resolved_subspecies.lower()
    
        # Populate the replacement dictionary
        substitution_dict[old] = new

    filtered_dict = {k: v for k, v in substitution_dict.items() if k != v}

    # Directly replace values in df_unmatched['gbif_taxonomy']
    df_unmatched['gbif_taxonomy'] = df_unmatched['gbif_taxonomy'].replace(filtered_dict, regex=True)
    df_unmatched_corrected = correct_inconsistent_subspecies(df_unmatched)
    
    return df_matched, df_unmatched_corrected, filtered_dict



def convert_tree_to_dataframe(tree, query_dataset, target_dataset, path, inat_dataset=None, index=False):
    """
    Converts a taxonomic tree into a pandas DataFrame, merges it with external datasets (NCBI, GBIF, iNaturalist),
    enriches it with synonyms, and saves the final DataFrame to a specified CSV file.

    Args:
    tree (object): The taxonomic tree to be converted.
    query_dataset (DataFrame): GBIF dataset containing taxonomic information.
    target_dataset (DataFrame): NCBI dataset containing taxonomic information.
    path (str): File path where the resulting CSV will be saved.
    inat_dataset (DataFrame, optional): iNaturalist dataset containing taxonomic information. Defaults to None.
    index (bool, optional): Whether to index the tree before conversion. Defaults to False.

    Returns:
    DataFrame: The cleaned, enriched DataFrame containing information from NCBI, GBIF, and iNaturalist.
    """
    # Step 1: Copy and optionally index the tree
    indexed_tree = tree[0].copy()
    if index:
        indexed_tree = index_tree(tree[0], include_name=True)

    # Step 2: Convert the tree to a DataFrame
    indexed_tree_data = tree_to_dataframe(indexed_tree, inat_dataset)
    df_indexed_tree = pd.DataFrame(indexed_tree_data).tail(-1)  # Exclude the root node

    # Step 3: Clean and cast IDs for NCBI, GBIF, and iNaturalist
    df_indexed_tree['ncbi_id'] = df_indexed_tree['ncbi_id'].fillna(-1).astype(int)
    df_indexed_tree['gbif_taxon_id'] = df_indexed_tree['gbif_taxon_id'].fillna(-1).astype(int)
    
    if inat_dataset is not None:
        if 'inat_taxon_id' in df_indexed_tree.columns:
            df_indexed_tree['inat_taxon_id'] = df_indexed_tree['inat_taxon_id'].fillna(-1).astype(int)

    '''
    # Remove rows where all IDs (NCBI, GBIF, iNaturalist) are -1
    df_indexed_tree = df_indexed_tree[
        ~((df_indexed_tree['ncbi_id'] == -1) &
          (df_indexed_tree['gbif_taxon_id'] == -1) &
          (df_indexed_tree['inat_taxon_id'] == -1)
         )
    ]
    '''
    
    # Step 4: Merge with GBIF and NCBI datasets
    target_dataset['ncbi_id'] = target_dataset['ncbi_id'].astype(int)
    query_dataset['taxonID'] = query_dataset['taxonID'].astype(int)

    merged_gbif = pd.merge(
        df_indexed_tree,
        query_dataset[['taxonID', 'canonicalName', 'gbif_taxonomy']],
        left_on='gbif_taxon_id',
        right_on='taxonID',
        how='left'
    )
    merged_ncbi = pd.merge(
        merged_gbif,
        target_dataset[['ncbi_id', 'ncbi_canonicalName', 'ncbi_target_string']],
        left_on='ncbi_id',
        right_on='ncbi_id',
        how='left'
    )

    # Step 5: Merge with iNaturalist dataset if provided
    if inat_dataset is not None:
        merged_inat = pd.merge(
            merged_ncbi,
            inat_dataset[['inat_taxon_id', 'inat_canonical_name', 'inat_taxonomy']],
            left_on='inat_taxon_id',
            right_on='inat_taxon_id',
            how='left'
        )
        final_dataset = merged_inat
    else:
        final_dataset = merged_ncbi

    # Step 6: Add synonym columns
    gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids = load_gbif_dictionary()
    ncbi_synonyms_names, ncbi_synonyms_ids = load_ncbi_dictionary()

    final_dataset_ = final_dataset.copy()
    
    final_dataset_['gbif_synonyms_names'] = final_dataset_['taxonID'].map(
        lambda x: '; '.join([name for name, _ in gbif_synonyms_ids_to_ids.get(x, [])])
    )

    final_dataset_['gbif_synonyms_ids'] = final_dataset_['taxonID'].map(
        lambda x: '; '.join([str(syn_id) for _, syn_id in gbif_synonyms_ids_to_ids.get(x, [])])
    )

    final_dataset_['ncbi_synonyms_names'] = final_dataset_['ncbi_id'].map(
        lambda x: '; '.join(ncbi_synonyms_ids.get(x, []))
    )
    final_dataset_['id'] = range(1, len(final_dataset_) + 1)  # Assign unique IDs

    # Step 7: Replace invalid values and clean the DataFrame
    final_dataset_ = final_dataset_.apply(lambda col: col.fillna(-1) if col.dtypes != 'object' else col.fillna("-1"))
    final_dataset_ = final_dataset_.replace([-1, '-1', ""], None)

    # Ensure the 'Index' column is converted to string
    final_dataset_['Index'] = final_dataset_['Index'].fillna('').astype(str)

    # Step 8: Dynamically select and rename columns
    selected_columns = ["id", "ncbi_id", "gbif_taxon_id", "ncbi_canonicalName", "canonicalName", "gbif_synonyms_names", "gbif_synonyms_ids", "ncbi_synonyms_names"]
    
    # Add columns dynamically based on conditions
    if inat_dataset is not None:
        selected_columns.extend(["inat_taxon_id", "inat_canonical_name"])
    
    if index:
        final_dataset_[['hierarchical_path', 'node_name']] = final_dataset_['Index'].str.extract(r'^([\d\.]+)\s*(.*)$', expand=True)
        selected_columns.extend(["hierarchical_path", "node_name"])
    
    # Rename the columns dynamically
    renamed_columns = {
        "id": "id",
        "hierarchical_path": "path",
        #"node_name": "node_name",
        "ncbi_id": "ncbi_taxon_id",
        "gbif_taxon_id": "gbif_taxon_id",
        "inat_taxon_id": "inat_taxon_id",
        "ncbi_canonicalName": "ncbi_canonical_name",
        "canonicalName": "gbif_canonical_name",
        "inat_canonical_name": "inat_canonical_name",
        "gbif_synonyms_ids": "gbif_synonyms_ids",
        "gbif_synonyms_names": "gbif_synonyms_names",
        "ncbi_synonyms_names": "ncbi_synonyms_names",

    }
    
    # Filter only existing columns to avoid KeyError
    existing_columns = [col for col in renamed_columns.keys() if col in final_dataset_.columns]
    final_dataset = final_dataset_[existing_columns]
    
    # Rename columns
    final_dataset_ = final_dataset.copy()
    final_dataset.rename(columns=renamed_columns, inplace=True)

    # Step 9: Clean synonyms and handle missing values
    final_results_cleaned = final_dataset.copy()
    final_results_cleaned['ncbi_synonyms_names'] = final_results_cleaned['ncbi_synonyms_names'].apply(clean_synonyms)
    
    if inat_dataset is not None:
        final_results_cleaned['inat_taxon_id'] = final_results_cleaned['inat_taxon_id'].astype('Int64')

    # Save the final DataFrame to CSV
    final_results_cleaned.to_csv(path, index=False)

    return final_results_cleaned



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
    



    