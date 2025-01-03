import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path:str)->pd.DataFrame:
    """Load the CSV data"""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data is successfully loaded")
        return df
    except FileNotFoundError as e:
        logger.error("File not found for loading the data: %s", e)
        raise
    except pd.errors.ParserError as e:
        logger.error("Error while parsing the data: %s", e)
        raise
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise
        
def train_model(X_train:np.ndarray, y_train:np.ndarray, params: dict)->RandomForestClassifier:
    """Train the model with the given parameters"""
    try:
        if X_train.shape[0]!= y_train.shape[0]:
            raise ValueError("The number of the sample must be same in both files")
        logger.debug("Initizialing the RandomForest model with the parameters:%s", params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
        logger.debug("Model training has been started with samples: %s", X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug("Model training has been completed")
        return clf
    
    except ValueError as e:
        logger.error("Value Error in the model training: %s",e)
        raise
    except Exception as e:
        logger.error("Error in the model training:%s", e)
        raise
    
def save_model(model, file_path:str)->None:
    """Save the model to the given file path"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug("File saved to: %s", file_path)
    except FileNotFoundError as e:
        logger.error("File not found for saving the model %s", e)
        raise 
    except Exception as e:
        logger.error("Error in the model saving: %s", e)
        raise
    
def main():
    try:
        params = load_params('params.yaml')['model_building']
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:,-1].values
        
        clf =train_model(X_train, y_train, params)
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)
        
    except Exception as e:
        logger.error("Failed to complete the model building process: %s", e)
        print(f"Error:{e}")
        
if __name__=='__main__':
    main()