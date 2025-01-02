import pandas as pd
import numpy as np
import os
import yaml 
from sklearn.model_selection import train_test_split


def load_params(filepath: str) -> float:
    try:
        with open(filepath, 'r') as file:
            params = yaml.safe_load(file)
        return params['data_collection']['test_size']   
    except Exception as e:
        raise Exception(f"An error occurred when trying to load the parameters: {e}")

# Load data
def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"An error occurred when trying to load the data: {e}")

# Split data
def split_data(df: pd.DataFrame, test_size: float) ->tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return train_test_split(df, test_size=test_size)
    except ValueError as e:
        raise ValueError(f"An error occurred when trying to split the data: {e}")

def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"An error occurred when trying to save the data: {e}")

def main():
    data_filepath = 'water_potability.csv'
    params_filepath = 'params.yaml'
    save_filepath = os.path.join('data', 'raw')
    try:
        df = load_data(data_filepath)
        test_size = load_params(params_filepath)
        train, test = split_data(df, test_size)

        os.makedirs(save_filepath, exist_ok=True)

        save_data(train, os.path.join(save_filepath, 'train.csv'))
        save_data(test, os.path.join(save_filepath, 'test.csv'))
    except Exception as e:
        raise Exception(f"An error occurred in the main function: {e}") 

if __name__ == '__main__':
    main()