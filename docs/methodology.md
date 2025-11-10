
# Methodology
This document details the end-to-end Drug–Gene 3D Association Modeling pipeline.

## Overview
The system unifies:
- Molecular-level 3D descriptor extraction (via RDKit)
- Protein sequence feature embedding (AAC, hydrophobicity, charge)
- Dual-branch neural architecture combining drug and gene features
- Feature-space learning for association discovery and drug repurposing
- Optional generative molecule design and docking integration.

## Training
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Early stopping on validation loss
- Metrics: Accuracy, Precision, Recall, F1, AUC

## Outputs
Artifacts include model weights, feature embeddings, evaluation metrics, and ranked candidate drug lists.
