import pandas as pd 
import numpy as np
import yaml
import pickle 
from sklearn.ensemble import RandomForestClassifier

def load_params(config_path: str) -> int:
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config['model_building']['n_estimators'], config['model_building']['max_depth']
    except Exception as e:  
        raise Exception(f'Failed to load configuration file: {e}')

def load_data(data_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        raise Exception(f'Failed to load data: {e}')

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X_train = data.drop(columns='Potability', axis = 1)
        y_train = data['Potability'] 
        return X_train, y_train
    except Exception as e:
        raise Exception(f'Failed to prepare data: {e}')
    
def train_model(X_train: pd.DataFrame, y_train: pd.Series, estimators: int, max_depth: int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=estimators, n_jobs=-1, max_depth=max_depth)
        clf.fit(X_train, y_train)
        return clf
    except Exception as e:
        raise Exception(f'Failed to train model: {e}')

def save_model(clf: RandomForestClassifier, model_path: str) -> None:
    try:
        pickle.dump(clf, open(model_path, 'wb'))
    except Exception as e:
        raise Exception(f'Failed to save model: {e}')

def main():
    params_path = 'params.yaml'
    data_path = 'data/processed/train_processed.csv'
    model_path = 'model.pkl'

    try:
        n_estimators, max_depth = load_params(params_path)
        data = load_data(data_path)
        X_train, y_train = prepare_data(data)
        clf = train_model(X_train, y_train, n_estimators, max_depth)
        save_model(clf, model_path)
    except Exception as e:
        raise Exception(f'Failed to execute main: {e}')


if __name__ == '__main__':
    main()