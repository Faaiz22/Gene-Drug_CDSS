
"""Small helpers to show molecules in Jupyter/Colab using py3Dmol (optional)."""
def view_mol_pdb_block(pdb_block):
    try:
        import py3Dmol
        view = py3Dmol.view(width=400, height=300)
        view.addModel(pdb_block, 'pdb')
        view.setStyle({'stick': {}})
        view.zoomTo()
        return view.show()
    except Exception:
        return None
