import os
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import string
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger("Data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(logs_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """Transform the text by converting it to the lowercase, tokenizing, removing the stopwords, punctuation and doing the steaming"""
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)    

def preprocess_df(df, text_column = 'text', target_column = 'target'):
    """Preprocess the dataframe by applying the transform_text function to the text column and encoding the target"""
    try:
        logger.debug("Start preprocessing the dataframe")
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Target columns has been encoded")
        
        df = df.drop_duplicates(keep = 'first')
        logger.debug("Duplicates has removed")
        
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed")
        return df
    
    except Exception as e:
        logger.error("Column not found %s", e)
        raise
    except Exception as e:
        logger.error("Error during text normalization %s", e)
        raise
    
def main(text_column = 'text', target_column = 'target'):
    """Main function to preprocess the dataframe"""
    try:
        train_data = pd.read_csv("./data/raw_data/train.csv")
        test_data = pd.read_csv("./data/raw_data/test.csv")
        logger.debug("Data loaded properly")
        
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)
        
        data_path = os.path.join("./data","interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"), index=False)
        logger.debug("Process data saved to: %s", data_path)
        
    except FileNotFoundError as e:
        logger.error("File not found %s", e)
        
    except pd.errors.EmptyDataError as e:
        logger.error("Dataframe is empty %s", e)
        
    except Exception as e:
        logger.error("Error during data preprocessing %s", e)
        print(f"Error:{e}")
        
if __name__=='__main__':
    main()