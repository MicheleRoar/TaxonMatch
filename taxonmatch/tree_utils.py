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