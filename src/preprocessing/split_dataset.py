
"""Stratified split utilities for 70/20/10 (train/val/test)."""
import numpy as np
from sklearn.model_selection import train_test_split

def stratified_split_indices(labels, test_size=0.1, val_size=0.2, random_state=42):
    # first split off test
    idx = np.arange(len(labels))
    trainval_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=labels, random_state=random_state)
    # then split train/val
    val_relative = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=val_relative, stratify=labels[trainval_idx], random_state=random_state)
    return train_idx, val_idx, test_idx
