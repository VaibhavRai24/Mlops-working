import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging 
from dvclive import live

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path:str):
    """Load the train model from a file"""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug("Model loaded from %s", file_path)
        return model
    except FileNotFoundError:
        logger.error("Model file not found at %s", file_path)
        raise
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise
        
def load_data(file_path:str)->pd.DataFrame:
    """Load the data from the CSV file"""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Error parsing data from loaded file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occur while loading the data: %s", e)
        raise
        
def evaluate_model(clf, X_test:np.ndarray, y_test:np.ndarray)->dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        logger.debug("Predictions: %s", y_pred)
        
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred,average='macro')

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise
    
def save_metrices(metrices:dict, file_path:str) ->None:
    """Save the evaluation metrices to a json file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        
        with open(file_path, 'w') as file:
            json.dump(metrices, file, indent=4 )
        logger.debug("Metirces saved to %s", file_path)
        
    except Exception as e:
        logger.error("Error saving metrices to file: %s", e)
        raise e
    
def main():
    try:
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)
        save_metrices(metrics, 'reports/metrics.json')
        
    except Exception as e:
        logger.error("Failed to complete the model evaluation process: %s", e)
        print(f"error: {e}")
        
if __name__=="__main__":
    main()