
"""IO utilities for saving/loading artifacts and zipping outputs."""
import json, zipfile, os
from pathlib import Path

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def make_zip(source_dir, zip_path):
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(source_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, source_dir)
                z.write(filepath, arcname)
    return zip_path
