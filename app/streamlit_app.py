import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import json
import torch
import base64
import io
from importlib import import_module
import os  # --- MODIFIED: Added for path joining and file checks
import requests  # --- MODIFIED: Added for downloading files

# --- MODIFIED: Robust pathing relative to this script file
# Get the directory where this script (streamlit_app.py) is located
SCRIPT_DIR = Path(__file__).parent
# Build the path to artifacts relative to this script's location
ARTIFACTS_DIR = (SCRIPT_DIR / "../artifacts").resolve()
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

# --- MODIFIED: URLs for large files (e.g., from a GitHub Release)
# --- YOU MUST UPDATE THESE URLS to point to your raw files!
# --- e.g., "https://github.com/your_user/your_repo/releases/download/v1.0/model.pt"
REMOTE_ARTIFACTS = {
    "pairwise_index_labels.csv": "YOUR_URL_TO_pairwise_index_labels.csv",
    "fusion_embeddings.npy": "YOUR_URL_TO_fusion_embeddings.npy",
    "features_drug.npy": "YOUR_URL_TO_features_drug.npy",
    "features_protein.npy": "YOUR_URL_TO_features_protein.npy",
    "model.pt": "YOUR_URL_TO_model.pt",
    "metrics.json": "YOUR_URL_TO_metrics.json",
    "metrics_test.json": "YOUR_URL_TO_metrics_test.json",
    # --- Add any other files that might be missing or in Git LFS
    # "fusion_umap_2d.npy": "YOUR_URL_TO_fusion_umap_2d.npy",
    # "loss_curve.png": "YOUR_URL_TO_loss_curve.png",
    # "metrics_comparison.png": "YOUR_URL_TO_metrics_comparison.png",
    # "Association_Model_3D.zip": "YOUR_URL_TO_Association_Model_3D.zip"
}


st.set_page_config(page_title="Drug–Gene 3D Association Explorer", layout="wide", initial_sidebar_state="expanded")

# --- MODIFIED: Helper function to download files
def download_file(filename, url):
    """
    Downloads a file from a URL to the local ARTIFACTS_DIR if it doesn't exist.
    """
    local_path = ARTIFACTS_DIR / filename
    if not local_path.exists():
        st.info(f"Downloading {filename}... (This happens once.)")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success(f"Downloaded {filename} successfully.")
        except Exception as e:
            st.error(f"Error downloading {filename}: {e}. Please check the URL in the script.")
            return None
    return local_path

# --- MODIFIED: Updated loader functions to use the downloader
@st.cache_data
def load_csv(filename):
    """
    Downloads the CSV if needed, then loads it.
    """
    local_path = download_file(filename, REMOTE_ARTIFACTS.get(filename))
    if local_path and local_path.exists():
        return pd.read_csv(local_path)
    return None

@st.cache_data
def load_npy(filename):
    """
    Downloads the NPY if needed, then loads it.
    """
    local_path = download_file(filename, REMOTE_ARTIFACTS.get(filename))
    if local_path and local_path.exists():
        return np.load(local_path)
    return None

@st.cache_resource
def load_model(filename):
    """
    Downloads the model if needed, then loads it.
    """
    local_path = download_file(filename, REMOTE_ARTIFACTS.get(filename))
    
    # minimal loader using src.models.dual_branch_net.DualBranchNet
    try:
        from src.models.dual_branch_net import DualBranchNet
    except Exception:
        class DualBranchNet:
            def __init__(self, *a, **k):
                pass
            def eval(self): pass
            def __call__(self, *a, **k): return np.array([0.5])
    
    model = DualBranchNet()
    
    if not (local_path and local_path.exists()):
        st.warning("Model file not found or failed to download. Using dummy model.")
        return model # Return dummy model

    try:
        import torch
        state = torch.load(local_path, map_location='cpu')
        try:
            model.load_state_dict(state)
        except Exception:
            try:
                model.load_state_dict(state.get('model_state_dict', state))
            except Exception:
                pass
    except Exception as e:
        st.error(f"Error loading model state: {e}. Using dummy model.")
    
    return model

def get_download_link_bytes(content: bytes, filename: str, label: str = "Download"):
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    return href

# Load artifacts (best-effort)
# --- MODIFIED: Changed to use new loader functions
pairwise_df = load_csv("pairwise_index_labels.csv")
fusion = load_npy("fusion_embeddings.npy")
drug_emb = load_npy("features_drug.npy")
prot_emb = load_npy("features_protein.npy")
model = load_model("model.pt")

metrics = {}
metrics_test = {}

# --- MODIFIED: Using download_file for JSONs too
metrics_path = download_file("metrics.json", REMOTE_ARTIFACTS.get("metrics.json"))
if metrics_path and metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)

metrics_test_path = download_file("metrics_test.json", REMOTE_ARTIFACTS.get("metrics_test.json"))
if metrics_test_path and metrics_test_path.exists():
    with open(metrics_test_path) as f:
        metrics_test = json.load(f)


# import components
try:
    from components.mol_viewer import mol_to_png_bytes
    from components.umap_plot import scatter_2d
    from components.docking_plot import docking_scatter
except Exception:
    mol_to_png_bytes = None
    scatter_2d = None
    docking_scatter = None

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview", "Data Explorer", "Latent Space", "Query", "Repurposing", "Generation", "Docking", "Diagnostics", "Downloads"
])

# Pages
if page == "Overview":
    st.title("Drug–Gene 3D Association Explorer — Overview")
    cols = st.columns(4)
    pairs = len(pairwise_df) if pairwise_df is not None else 0
    genes = pairwise_df['Gene_ID'].nunique() if pairwise_df is not None else 0
    drugs = pairwise_df['Drug_ID'].nunique() if pairwise_df is not None else 0
    cols[0].metric("Pairs", pairs)
    cols[1].metric("Genes", genes)
    cols[2].metric("Drugs", drugs)
    cols[3].metric("AUC (val/test)", f"{metrics.get('auc','-')}/{metrics_test.get('auc','-')}")
    st.markdown("""This app loads artifacts produced by the training pipeline. Use the sidebar to explore the dataset and models.""")

elif page == "Data Explorer":
    st.header("Data Explorer")
    if pairwise_df is None:
        st.warning("pairwise_index_labels.csv not found or failed to load.")
    else:
        st.dataframe(pairwise_df.head(200))
        csv = pairwise_df.to_csv(index=False).encode()
        st.markdown(get_download_link_bytes(csv, "pairwise_preview.csv", "Download preview"), unsafe_allow_html=True)

elif page == "Latent Space":
    st.header("Latent Space Viewer")
    if fusion is None or pairwise_df is None:
        st.warning("Fusion embeddings or pairwise index not available.")
    else:
        # --- MODIFIED: Added downloader for this optional file
        coords_path_name = "fusion_umap_2d.npy"
        coords_path = download_file(coords_path_name, REMOTE_ARTIFACTS.get(coords_path_name))
        
        if coords_path and coords_path.exists():
            coords = np.load(coords_path)
        else:
            try:
                import umap
                st.info("Calculating UMAP... (This happens once.)")
                coords = umap.UMAP(n_components=2, random_state=42).fit_transform(fusion)
            except Exception:
                from sklearn.decomposition import PCA
                st.info("Calculating PCA... (This happens once.)")
                coords = PCA(n_components=2).fit_transform(fusion)
        if scatter_2d:
            fig = scatter_2d(coords, pairwise_df, color_by=st.selectbox("Color by", ["Association_Label","Gene_ID","Drug_ID"]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Install app components to enable interactive plotting.")

elif page == "Query":
    st.header("Gene–Drug Query")
    st.write("Provide Gene_ID and Drug_SMILES or choose from dataset to compute model score.")
    if pairwise_df is None:
        st.warning("Dataset required.")
    else:
        gene = st.selectbox("Gene_ID", options=["--"] + sorted(pairwise_df['Gene_ID'].astype(str).unique().tolist()))
        drug = st.selectbox("Drug_ID (optional)", options=["--"] + sorted(pairwise_df['Drug_ID'].astype(str).unique().tolist()))
        smi = st.text_input("Or enter SMILES (overrides Drug_ID)")
        if st.button("Predict"):
            # minimal prediction: map to embeddings if available
            idx = None
            if drug != "--":
                rows = pairwise_df[pairwise_df['Drug_ID'].astype(str)==drug]
                if not rows.empty:
                    idx = rows.index[0]
            if smi and mol_to_png_bytes:
                st.image(mol_to_png_bytes(smi))
            if drug_emb is not None and prot_emb is not None and idx is not None:
                import torch
                d = torch.tensor(drug_emb[idx:idx+1], dtype=torch.float32)
                p = torch.tensor(prot_emb[idx:idx+1], dtype=torch.float32)
                try:
                    prob = float(model(d,p).cpu().numpy().ravel()[0])
                except Exception:
                    prob = 0.5
                st.metric("Predicted association", f"{prob:.3f}")
            else:
                st.info("Embeddings or model not available for concrete prediction.")

elif page == "Repurposing":
    st.header("Repurposing Engine")
    if pairwise_df is None or fusion is None:
        st.warning("Need pairwise table and fusion embeddings.")
    else:
        gene = st.selectbox("Choose gene", options=["--"] + sorted(pairwise_df['Gene_ID'].astype(str).unique().tolist()))
        k = st.slider("Top-K candidates", 5, 200, 25)
        exclude_known = st.checkbox("Exclude known positives", value=True)
        if st.button("Run"):
            import numpy as np
            import pandas as pd
            idxs = pairwise_df.index[pairwise_df['Gene_ID'].astype(str)==gene].tolist()
            if not idxs:
                st.error("Gene not present or no entries.")
            else:
                pos = [i for i in idxs if pairwise_df.loc[i,'Association_Label']==1]
                centroid = fusion[pos].mean(axis=0) if pos else fusion[idxs].mean(axis=0)
                sims = np.dot(fusion, centroid) / (np.linalg.norm(fusion,axis=1)*np.linalg.norm(centroid)+1e-9)
                dfc = pairwise_df.copy()
                dfc['similarity'] = sims
                if exclude_known:
                    known = set(pairwise_df.loc[pairwise_df['Gene_ID']==gene].loc[lambda d: d['Association_Label']==1, 'Drug_ID'].tolist())
                    dfc = dfc[~dfc['Drug_ID'].isin(known)]
                out = dfc.sort_values('similarity', ascending=False).head(k)
                st.dataframe(out[['Gene_ID','Drug_ID','similarity','Association_Label']].reset_index(drop=True))
                st.markdown(get_download_link_bytes(out.to_csv(index=False).encode(), f"repurposing_{gene}.csv", "Download"), unsafe_allow_html=True)

elif page == "Generation":
    st.header("Generative Design (Scaffold)")
    st.write("""This is a scaffold: retrieval + conservative perturbation is used as a baseline generator.
             For full generator, integrate a ConditionalVAE/Transformer in src/models.""")
    st.info("Generation is offline — use notebook/pipelines to run heavy operations.")

elif page == "Docking":
    st.header("Docking Integration")
    uploaded = st.file_uploader("Upload docking CSV (SMILES,Docking_Energy)", type=['csv','tsv','json'])
    if uploaded:
        try:
            dock_df = pd.read_csv(uploaded) if str(uploaded.name).endswith('.csv') else pd.read_json(uploaded)
            st.dataframe(dock_df.head(20))
            # --- MODIFIED: This part is tricky, as it relies on finding a file.
            # You may need to adjust this logic depending on where 'Final_Ranked_Candidates' comes from.
            # For now, it will just look in the local artifacts dir.
            final_csvs = list(ARTIFACTS_DIR.glob('Final_Ranked_Candidates_*.csv'))
            if final_csvs:
                final = pd.read_csv(final_csvs[0])
                merged = final.merge(dock_df[['SMILES','Docking_Energy']], on='SMILES', how='left')
                merged['Docking_Energy'] = merged['Docking_Energy'].fillna(merged['Docking_Energy'].mean())
                merged['Docking_Norm'] = (merged['Docking_Energy'].max() - merged['Docking_Energy']) / (merged['Docking_Energy'].max() - merged['Docking_Energy'].min()+1e-9)
                merged['Final_Integrated_Score'] = 0.4*merged['Final_Score'] + 0.4*merged['Docking_Norm'] + 0.2*merged['Pred_Assoc']
                st.dataframe(merged.sort_values('Final_Integrated_Score', ascending=False).head(30))
            else:
                st.warning('No Final_Ranked_Candidates CSV found in artifacts.')
        except Exception as e:
            st.error(f"Failed to parse docking file: {e}")

elif page == "Diagnostics":
    st.header("Diagnostics")
    # --- MODIFIED: Added downloader for these optional image files
    loss_curve_path = download_file("loss_curve.png", REMOTE_ARTIFACTS.get("loss_curve.png"))
    if loss_curve_path and loss_curve_path.exists():
        st.image(str(loss_curve_path))
        
    metrics_comp_path = download_file("metrics_comparison.png", REMOTE_ARTIFACTS.get("metrics_comparison.png"))
    if metrics_comp_path and metrics_comp_path.exists():
        st.image(str(metrics_comp_path))
        
    st.write('Metrics (val/train/test):', metrics, metrics_test)

elif page == "Downloads":
    st.header("Downloads")
    # --- MODIFIED: Added downloader for the zip file
    zip_path = download_file("Association_Model_3D.zip", REMOTE_ARTIFACTS.get("Association_Model_3D.zip"))
    if zip_path and zip_path.exists():
        st.markdown(get_download_link_bytes(zip_path.read_bytes(), 'Association_Model_3D.zip', 'Download Zip'), unsafe_allow_html=True)
    else:
        st.info('Run packaging script to create zip (pipelines/package_artifacts.py) and upload it to your release.')

# Footer
st.sidebar.markdown('---')
st.sidebar.write('Drug_Gene_CDSS — Research demo')
