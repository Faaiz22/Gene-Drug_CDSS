
"""Zip artifacts directory into Association_Model_3D.zip"""
import os
from src.utils.io_utils import make_zip

def package(artifacts_dir='artifacts', out_zip='Association_Model_3D.zip'):
    return make_zip(artifacts_dir, out_zip)

if __name__ == '__main__':
    print('Run package() programmatically to zip artifacts.')
