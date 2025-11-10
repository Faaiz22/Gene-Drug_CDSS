# Drug_Gene_CDSS
A unified 3D gene–drug association and clinical decision support system integrating molecular modeling,
deep learning, and generative chemistry.

## Features
- Dual-branch neural network for drug–gene association
- 3D molecular descriptor extraction via RDKit
- Protein sequence embeddings (AAC, charge, hydrophobicity)
- Generative molecule design and docking integration
- Interactive Streamlit visualization app

## Quick Start
```bash
conda env create -f environment.yml
conda activate drug_gene_cdss
streamlit run app/streamlit_app.py
```

## Folder Overview
See docs/methodology.md for methodology and architecture details.
