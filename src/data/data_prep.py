import pandas as pd 
import numpy as np 
import os 

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"An error occurred when trying to load the data: {e}") 


def fill_missing_with_median(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
        return df
    except Exception as e:  
        raise Exception(f"An error occurred when trying to fill missing values: {e}")

def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"An error occurred when trying to save the data: {e}") 


def main():
    raw_data_path = os.path.join('data', 'raw')
    processed_data_path = os.path.join('data', 'processed')

    train_data = load_data(os.path.join(raw_data_path, 'train.csv'))
    test_data = load_data(os.path.join(raw_data_path, 'test.csv'))

    train_processed_data = fill_missing_with_median(train_data)
    test_processed_data = fill_missing_with_median(test_data)

    os.makedirs(processed_data_path, exist_ok=True)

    save_data(train_processed_data, os.path.join(processed_data_path, 'train_processed.csv'))
    save_data(test_processed_data, os.path.join(processed_data_path, 'test_processed.csv'))


if __name__ == '__main__':  
    main()

