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
    __word2vec_model = None
    __X = None
    __y = None
    __X_trian = None
    __y_train = None
    __y_test = None
    __X_test = None
    __model = None

    def __init__(self) -> None:
        pass

    def read_data(self, file_name: str) -> None:
        """
        Reads the dataset, if the file path mentioned is not found an exception is raised.
        """
        try:
            print("[I/O] Reading csv file: "+file_name)
            self.__data_frame = pd.read_csv(file_name)
            self.__data_frame.dropna()
            print("[RESULT] Read file successfully, and initialized the data frame.")
        except Exception as e:
            print("[ERR] The following error occured while trying to read data from a csv file: "+str(e))

    def get_five(self) -> None:
        """
        Prints first five rows of the data to view the current state of each feature.
        """
        try:
            print("\n\n[DATA] Here are five samples of current dataset: ")
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
            print("[I/O] Please wait this might take some time.")
            self.__data_frame['preprocessed_text'] = self.__data_frame["comment_text"].apply(self.preprocess)
            print("[DATA] Data cleaned successfully, the new feature created is 'preprocessed_text'.")
        except Exception as e:
            print("[ERR] the following error occured while trying to clean the data: "+str(e))

    def train_word2vec_model(self, embedding_size = 100) -> None:
        """
        Using the new feature that was previously created by method 'clean_data'.
        We shall now train a word2vec model.
        """
        try:
            print("[Word2Vec] Training a Word2Vec Model with embedding size: "+str(embedding_size))
            print("[I/O] Please wait this might take some time.")
            self.__word2vec_model = Word2Vec(self.__data_frame['preprocessed_text'], vector_size=embedding_size, window=5, min_count=2, workers=4)
            print("[Word2Vec] New instance of the model has been trained successfully")
            print("Here is the object reference: "+str(self.__word2vec_model))
        except Exception as e:
            print("[ERR] The following error occured while trying to train a Word2Vec model: "+str(e))

  
    def text_to_embedding(self, text, embedding_size = 100) -> list:
        """
        This method takes in a list of words within an sentence that has been cleaned.
        Then generates a embedding of fixed size for each word withing numpy mean and word2vec model that was trained previously.
        """
        try:
            if text is None:
                return
            words = [word for word in text if word in self.__word2vec_model.wv]
            if words:
                return np.mean(self.__word2vec_model.wv[words], axis=0)
            else:
                return np.zeros(embedding_size)
        except Exception as e:
            print("[ERR] The following error occured while trying to generate an embedding for a comment: "+str(e))
        
    def create_embeddings(self) -> None:
        """
        Creates a new feature 'embedding' within the data frame for each comment.
        This is achieved by the help of method text_to_embeddings that will generate an embedding for each comment.
        """
        try:
            print("[Word2Vec] Creating word embeddings for each comment within a new feature 'embedding'.")
            print("[I/O] Please wait this might take longer than expected!")

            self.__data_frame['embedding'] = self.__data_frame['preprocessed_text'].apply(self.text_to_embedding)
            print("[Word2Vec] New feature 'embedding' created successfully!")
            self.get_five()
        except Exception as e:
            print("[ERR] The following error occured while creating embeddings: " +str(e))

    def split_data(self) -> None:
        """
        Creates Training set and Test set out of Dataframe.
        """
        try:
            print("[DATA] Performing the data split.")
            self.__X = np.stack(self.__data_frame['embedding'].values)
            self.__y = self.__data_frame['toxic'].values

            self.__X_trian, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size= 0.3, random_state=42)

            print("[DATA] Training set and Test set have been extracted successfully from the dataframe!")
        except Exception as e:
            print("[ERR] The following error occured while trying to split the dataframe: "+str(e))

    def get_model_metrics(self) -> None:
        """
        Prints the metrics for the newly trained MLP classifier.
        """
        try:
            pass
        except Exception as e:
            print("[ERR] The following error occured while trying to compute classfication metrics of the MLP model: "+str(e))

    def train_MLP_model(self, path_to_train_file: str, num_layers = 2) -> MLPClassifier:
        """
        This method, trains a Multi-layer preceptron and then returns it.
        This training is performed using Embeddings on comment_text of user.
        """
        try:
            self.read_data(path_to_train_file) #read the csv file.
            self.get_five() #see the data.
            self.clean_data()
            self.get_five() #show the updated data frame.
            
            self.train_word2vec_model() #train the word2vec model.
            self.create_embeddings() #create word embeddings.

            self.split_data() #creates training and test set for model classification metrics.

            print("[MLP] Creating a new instance of a Multi Layer Perceptron!")
            self.__model = MLPClassifier(hidden_layer_sizes=(64, 32),max_iter=1000, activation='relu', solver='adam', random_state=42)
            print("[MLP] A new instance for Multi Layer Perceptron has been created with the following hyper params: ")
            print("[MLP] Size of Hidden Layers: "+str(num_layers))
            print("[MLP] Max Iterations: 1000")
            print("[MLP] Activation Function: ReLu ( x if x > 0 else 0 )")
            print("[MLP] Solver: Adam.")

            print("[MLP] Training this new instance of MLP, please wait this might take more than a minute.")
            self.__model.fit(self.__X_trian, self.__y_train)
            print("[MLP] Object reference for the trained and newly fitted MLP model: "+str(self.__model))

        except Exception as e:
            print("[ERR] The following error occured while trying to train an MLP classifier: "+str(e))

Obj = MLP_Classifier()
Obj.train_MLP_model("./datasets/train.csv")