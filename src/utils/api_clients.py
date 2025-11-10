
"""Small wrappers for external APIs (NCBI, PubChem) with simple retries."""
import requests, time

def simple_get(url, params=None, headers=None, retries=3, backoff=1.0):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            time.sleep(backoff * (i+1))
    return None
