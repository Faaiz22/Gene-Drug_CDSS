
from src.preprocessing.cleaning import create_association_label
import pandas as pd

def test_label_creation():
    df = pd.DataFrame({'Association': ['associated', 'none']})
    df2 = create_association_label(df)
    assert 'Association_Label' in df2.columns
