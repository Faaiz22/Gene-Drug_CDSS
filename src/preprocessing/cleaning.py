
"""Cleaning helpers: dedupe, filter, association label creation."""
import pandas as pd

def create_association_label(df, assoc_col="Association"):
    # Expect string values like 'associated' or 'not associated' -> binary 1/0
    df = df.copy()
    if assoc_col in df.columns:
        df['Association_Label'] = df[assoc_col].astype(str).str.contains('assoc', case=False).astype(int)
    else:
        # fallback: if there is a 'score' column
        df['Association_Label'] = df.get('Association_Label', 0).astype(int)
    return df

def keep_genes_with_min_drugs(df, gene_col="Gene_ID", min_drugs=10):
    counts = df[df['Association_Label']==1][gene_col].value_counts()
    keep = counts[counts >= min_drugs].index
    return df[df[gene_col].isin(keep)].reset_index(drop=True)
