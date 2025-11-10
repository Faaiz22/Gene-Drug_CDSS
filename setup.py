from setuptools import setup, find_packages

setup(
    name='Drug_Gene_CDSS',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Unified 3D Gene–Drug Association Clinical Decision Support System',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch', 'pandas', 'numpy', 'scikit-learn', 'biopython',
        'rdkit-pypi', 'tqdm', 'umap-learn', 'matplotlib', 'seaborn',
        'plotly', 'streamlit', 'py3Dmol', 'selfies'
    ],
    python_requires='>=3.8',
)
