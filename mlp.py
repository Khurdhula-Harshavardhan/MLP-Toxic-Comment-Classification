import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


class MLP_Classifier():
    __model = None
    __file_handler = None
    __data_frame = None
    def __init__(self) -> None:
        pass

    def read_data(self, file_name: str) -> None:
        """
        Reads the dataset, if the file path mentioned is not found an exception is raised.
        """
        try:
            print("[I/O] Reading csv file: "+file_name)
            self.__data_frame = pd.read_csv(file_name)
            print("[RESULT] Read file successfully, and initialized the data frame.")
        except Exception as e:
            print("[ERR] The following error occured while trying to read data from a csv file: "+str(e))

    def get_five(self) -> None:
        """
        Prints first five rows of the data to view the current state of each feature.
        """
        try:
            print("[DATA] Here are five samples of current dataset: ")
            print(self.__data_frame.head())
        except Exception as e:
            print("[ERR] Something went wrong while trying to print data!")

    def preprocess(self, text):
        try:
            return simple_preprocess(text, deacc=True, min_len=2, max_len=15)
        except Exception as e:
            print("[ERR] The following error occured while trying to preprocess the data!: "+str(e) )
    
    def clean_data(self) -> None:
        """
        Applies the method preprocess to the data frame to create the tokens that are necessary for creating embeddings.
        """
        try:
            print("[DATA] Cleaning this data and Creating a new feature that contains all tokens.")
            self.__data_frame['preprocessed_text'] = self.__data_frame["comment_text"].apply(self.preprocess)
            print("[DATA] Data cleaned successfully, the new feature created is 'preprocessed_text'.")
        except Exception as e:
            print("[ERR] the following error occured while trying to clean the data: "+str(e))

    def train_MLP_model(self, path_to_train_file: str, num_layers = 2) -> MLPClassifier:
        """
        This method, trains a Multi-layer preceptron and then returns it.
        This training is performed using Embeddings on comment_text of user.
        """
        try:
            self.read_data(path_to_train_file) #read the csv file.
            self.get_five() #see the data.

            

        except Exception as e:
            print("[ERR] The following error occured while trying to train an MLP classifier: "+str(e))

Obj = MLP_Classifier()
Obj.train_MLP_model("./datasets/train.csv")