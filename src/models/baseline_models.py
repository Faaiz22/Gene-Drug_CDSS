
"""Simple baseline models (sklearn wrappers)."""
from sklearn.linear_model import LogisticRegression
import numpy as np

class LogisticBaseline:
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]
