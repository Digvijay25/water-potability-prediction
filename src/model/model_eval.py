import pandas as pd 
import numpy as np 
import json 
import pickle 
from dvclive import Live
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found in the path: {filepath}")

def prepare_data(data: pd.DataFrame) -> tuple:
    X = data.iloc[:,0:-1]
    y = data.iloc[:,-1].values
    return X, y

def load_model(filepath: str) -> any:
    try:
        return pickle.load(open(filepath, 'rb'))
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found in the path: {filepath}")

def evaluate_model(model: any, X_test: pd.DataFrame, y_test: np.array) -> dict:
    try:

        params = yaml.safe_load(open('params.yaml', 'r'))
        test_size = params['data_collection']['test_size']
        n_estimators = params['model_building']['n_estimators']
        max_depth = params['model_building']['max_depth']

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        with Live(save_dvc_exp = True) as live:
            live.log("accuracy", acc)
            live.log("precision", pre)
            live.log("recall", recall)
            live.log("f1-score", f1)

            live.log("test_size", test_size)
            live.log("n_estimators", n_estimators)  
            live.log("max_depth", max_depth)

        metrics = {
            'accuracy': acc,
            'precision': pre,
            'recall': recall,
            'f1-score': f1
        }
        return metrics
    except Exception as e:
        raise Exception(f"An error occured: {e}")
    

def save_metrics(metrics: dict, filepath: str) -> None:
    try:
        with open(filepath, 'w') as file:
            json.dump(metrics, file, indent=4)  
    except Exception as e:
        raise Exception(f"An error occured: {e}")
    

def main():
    try:
        test_data = load_data('data/processed/test_processed_mean.csv')
        X_test, y_test = prepare_data(test_data)
        model = load_model('models/model.pkl')
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        print(f"An error occured while evaluating the model: {e}")


if __name__ == '__main__':
    main() 