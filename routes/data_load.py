import pandas as pd
import os

def load_data(data_path='archive/train.csv', nrows=100000):
    """
    Load the training data from CSV file, with optional row limit.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, nrows=nrows)
    return df

def load_test_data(data_path='archive/test.csv', nrows=10000):
    """
    Load the test data from CSV file, with optional row limit.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data file not found: {data_path}")

    df = pd.read_csv(data_path, nrows=nrows)
    return df