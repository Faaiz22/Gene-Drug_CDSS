
"""Robust FASTA fetcher using BioPython Entrez with caching.

Functions:
 - fetch_protein_fasta_for_gene(ncbi_gene_id, email, api_key=None, cache_path=None)
"""
import os, json, time
from Bio import Entrez, SeqIO
from pathlib import Path

def fetch_protein_fasta_for_gene(ncbi_gene_id, gene_name=None, email=None, api_key=None, cache_path=None, max_aa=1000):
    if email is None:
        raise ValueError("Provide an email for NCBI Entrez.")
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    cache_path = Path(cache_path) if cache_path else None
    cache = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    key = str(ncbi_gene_id)
    if key in cache:
        return cache[key]
    try:
        # attempt to search protein db for refseq entries
        query = f"{gene_name if gene_name else ncbi_gene_id}[gene] AND srcdb_refseq[PROP]"
        handle = Entrez.esearch(db="protein", term=query, retmax=1)
        rec = Entrez.read(handle)
        ids = rec.get('IdList', [])
        handle.close()
        if not ids:
            cache[key] = ''
            if cache_path:
                cache_path.write_text(json.dumps(cache))
            return ''
        prot_id = ids[0]
        fh = Entrez.efetch(db='protein', id=prot_id, rettype='fasta', retmode='text')
        rec = SeqIO.read(fh, 'fasta')
        fh.close()
        seq = str(rec.seq)[:max_aa]
        cache[key] = seq
        if cache_path:
            cache_path.write_text(json.dumps(cache))
        # rate limit
        time.sleep(0.12)
        return seq
    except Exception as e:
        cache[key] = ''
        if cache_path:
            cache_path.write_text(json.dumps(cache))
        return ''

