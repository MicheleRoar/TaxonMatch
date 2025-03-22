# 🌿 TaxonMatch: Integrating Taxonomic Data from GBIF, NCBI, iNaturalist, PaleoDB, IUCN and other sources

## 📌 Introduction
**TaxonMatch** is a Python framework designed to **integrate, clean, and analyze taxonomic data** from **GBIF**, **NCBI**, **iNaturalist**, **PaleoDB**, and **IUCN**. It enhances taxonomic consistency across biodiversity datasets, simplifies taxonomic name matching, and enables the generation of **phylogenetic trees** based on consolidated data.

### 🔍 Main Features
- 📥 **Download and clean taxonomic datasets** from GBIF, NCBI, iNaturalist, PaleoDB, and IUCN.
- 🔗 **Taxonomic name matching** to identify synonyms and discrepancies.
- 🌳 **Generate phylogenetic trees** from consolidated taxonomic data.
- 🦴 **Analyze fossil taxa** and identify their closest living relatives.
- 🌍 **Assign conservation status** to species using IUCN data.

---

## ⚙ Installation & Setup

### 🐍 Create a clean Conda environment (recommended)

```bash
conda create -n taxonmatch-env python=3.10 -y
conda activate taxonmatch-env

```

### 📦 Install TaxonMatch

```bash
#Install the latest version directly from GitHub:
pip install git+https://github.com/MicheleRoar/TaxonMatch.git
```

### 📓 (Optional) Run example notebooks
```bash
#Install Jupyter and optional plotting libraries:
pip install notebook ipython ipywidgets
jupyter notebook
```

### ⚠️ macOS + XGBoost fix (Apple Silicon only)
```bash
#If you're on macOS with M1/M2/M3 and XGBoost fails to load due to libomp.dylib
arch -arm64 brew install libomp
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
#To make this permanent, add to your shell config (~/.zshrc or ~/.bash_profile):
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
#Then run:
source ~/.zshrc  # or source ~/.bash_profile
```

## 🚀 Usage & Workflow

### 1️⃣ Download Taxonomic Datasets
```bash
import taxonmatch as txm

# Download GBIF and NCBI datasets
gbif_dataset = txm.download_gbif_taxonomy()
ncbi_dataset = txm.download_ncbi_taxonomy()
```

### 2️⃣ Taxonomic Name Matching (GBIF vs NCBI)
```bash
# Select a specific clade (example: Apidae)
gbif_apidae, ncbi_apidae = txm.select_taxonomic_clade("Apidae", gbif_dataset, ncbi_dataset)

# Perform taxonomic matching
matched_df, unmatched_df, possible_typos_df = txm.match_dataset(gbif_apidae, ncbi_apidae)
```

### 3️⃣ Generate a Phylogenetic Tree
```bash
tree = txm.generate_taxonomic_tree(matched_df, unmatched_df)
txm.print_tree(tree, root_name="Apidae")
txm.save_tree(tree, "taxon_tree.txt")
```

### 4️⃣ Add Conservation Status Using IUCN Data
```bash
df_with_iucn_status = txm.add_iucn_status_column(matched_df)
df_with_iucn_status[df_with_iucn_status.iucnRedListCategory.isin(['ENDANGERED', 'CRITICALLY_ENDANGERED', 'VULNERABLE'])]
```

### 5️⃣ Identify Closest Living Relatives for Fossil Species
```bash
paleodb_dataset = txm.download_gbif_taxonomy(source="paleodb")
a3cat = txm.download_ncbi_taxonomy(source="a3cat")

query = "Arthropoda;Insecta;Hymenoptera;Formicidae;Formica;Formica seuberti"
txm.find_top_n_similar(query, a3cat, n_neighbors=4)
```

## 📈 Visualization
### Distribution of Conservation Status
```bash
import matplotlib.pyplot as plt

conservation_counts = df_with_iucn_status['iucnRedListCategory'].value_counts()
plt.figure(figsize=(10, 6))
conservation_counts.plot(kind='bar', color="blue", edgecolor='black')
plt.title('Distribution by Conservation Status')
plt.ylabel('Number of Species')
plt.xlabel('Category')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## License

This project is licensed under the MIT License.

## 🤝 Contributing & Maintenance

If you want to contribute to **TaxonMatch**:

- **Open an issue** to report bugs or suggest improvements.
- **Create a pull request** with a clear description of your changes.
- **Follow the coding guidelines** and ensure all modifications are tested.

For questions or support, contact michele.leone@unil.ch.


## License

This project is licensed under the MIT License.




