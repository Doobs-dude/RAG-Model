import os
import pandas as pd

def load_raw_dataset(csv_path: str = "data/arXiv_scientific_dataset.csv") -> pd.DataFrame:
    #Loads the raw dataset into a DataFrame
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at {csv_path}")
    return pd.read_csv(csv_path)

def save_metadata(df: pd.DataFrame, save_path: str = "vector_store/arxiv_full_metadata.pkl") -> None:
    #Saves metadata into a pickle file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_pickle(save_path)

def load_metadata(pickle_path: str = "vector_store/arxiv_full_metadata.pkl") -> pd.DataFrame:
    #Loads metadata into a pickle file
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Metadata file not found at {pickle_path}")
    return pd.read_pickle(pickle_path)