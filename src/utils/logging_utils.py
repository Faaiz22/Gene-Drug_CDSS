
"""Small logging setup helper using python logging config file."""
import logging, logging.config, yaml
from pathlib import Path

def setup_logging(config_path=None):
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(level=logging.INFO)
    return logging.getLogger('DrugGeneCDSS')
