import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok= True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler_loger = logging.StreamHandler()
console_handler_loger.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler_logger = logging.FileHandler(log_file_path)
file_handler_logger.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler_loger.setFormatter(formatter)
file_handler_logger.setFormatter(formatter)

logger.addHandler(console_handler_loger)
logger.addHandler(file_handler_logger)

def load_data(data_url:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from the %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Error parsing the data from the %s", data_url)
        raise e
    except Exception as e:
        logger.error(" Unexpected Error loading the data from the %s", data_url)
        raise e
    
def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace= True)
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug("DATA PREPROCESSING IS COMPLETED")
        return df
    except KeyError as e:
        logger.error("Error during data preprocessing: %s", e)
        raise e
    except Exception as e:
        logger.error("Unexpected Error during data preprocessing: %s", e)
        raise e
    
def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame, data_path:str)-> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw_data')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug("TRAINING AND TESTING DATA ARE SAVED TO THE RAW DATA PATH", raw_data_path)
        
    except Exception as e:
        logger.error("Unexpected Error saving the data to the raw data path: %s", e)
        raise e
    
def main():
    try:
        test_size = 0.2
        data_url = "https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv"
        df = load_data(data_url=data_url)
        final_df = preprocess_data(df=df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state= 2)
        save_data(train_data=train_data, test_data=test_data, data_path='./data')
    except Exception as e:
        logger.error("Error while performing the main function:%s ", e)
        print(f"Error:{e}")
        
if __name__ =='__main__':
    main()