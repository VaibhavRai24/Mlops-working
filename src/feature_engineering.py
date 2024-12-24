import pandas as pd
import os
import logging
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)-> pd.DataFrame:
    """Load the data from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        df.fillna('0', inplace= True)
        logger.debug("Data has loaded and NANS has filled from :%s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Error while loading data from file: %s", file_path)
        raise e
    except Exception as e:
        logger.error("Unexcepted error has been occured while loadind the data")
        raise e
    
def apply_tfidf(train_data:pd.DataFrame, test_data:pd.DataFrame, max_features: int) -> tuple:
    """apply the tfidf to the data"""
    try:
        vectorizer = TfidfVectorizer(max_features= max_features)
        X_train = train_data['text'].values
        X_test = test_data['text'].values
        y_train = train_data['target'].values
        y_test = test_data['target'].values
        
        X_train_bow  = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        
        logger.debug("Tfidf has been applied and the data has been transformed")
        return train_df, test_df
    
    except Exception as e:
        logger.error("Unexcepted error has been occured while applying the tfidf")
        raise
    
def save_data(df:pd.DataFrame, file_path:str) ->None:
    """Save the data to a CSV file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to :%s', file_path)
    except Exception as e:
        logger.error("Unexcepted error has been occured while saving the data")
        raise
    
def main():
    try:
        max_features = 100
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)
        
        save_data(train_data, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_data, os.path.join("./data", "processed", "test_tfidf.csv"))
        
    except Exception as e:
        logger.error("Failed to complete the feature engineering problem")
        print(f"Error:{e}")
        
if __name__=='__main__':
    main()