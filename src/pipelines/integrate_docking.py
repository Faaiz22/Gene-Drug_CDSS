
"""Merge external docking CSV with ranked candidates and compute integrated score."""
import pandas as pd, numpy as np

def integrate(final_ranked_csv, docking_csv, out_csv):
    fr = pd.read_csv(final_ranked_csv)
    dock = pd.read_csv(docking_csv)
    # sanitize columns
    if 'SMILES' not in dock.columns:
        raise ValueError('Docking CSV must contain SMILES column')
    merged = fr.merge(dock[['SMILES','Docking_Energy']], on='SMILES', how='left')
    merged['Docking_Energy'] = merged['Docking_Energy'].fillna(merged['Docking_Energy'].mean())
    merged['Docking_Norm'] = (merged['Docking_Energy'].max() - merged['Docking_Energy']) / (merged['Docking_Energy'].max() - merged['Docking_Energy'].min() + 1e-9)
    merged['Final_Integrated_Score'] = 0.4*merged['Final_Score'] + 0.4*merged['Docking_Norm'] + 0.2*merged['Pred_Assoc']
    merged.sort_values('Final_Integrated_Score', ascending=False).to_csv(out_csv, index=False)
    return out_csv
