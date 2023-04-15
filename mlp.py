import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc


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

    

    def plot_confusion_matrix(self, y_true, y_pred, labels=['Not Toxic', 'Toxic'], title='Confusion Matrix'):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.show()

    def get_model_metrics(self) -> None:
        """
        Prints the metrics for the newly trained MLP classifier.
        """
        try:
            print("[METRICS] Here are the classification metrics for the model: ")
            y_pred = self.__model.predict(self.__X_test)
            #self.plot_confusion_matrix(y_true=self.__y_test, y_pred=y_pred)
            print(classification_report(self.__y_test, y_pred=y_pred))
        except Exception as e:
            print("[ERR] The following error occured while trying to compute classfication metrics of the MLP model: "+str(e))

    def get_layers(self, number: int) -> tuple:
        """
        Gets the tuple that serves as layer size.
        """
        try:
            layers= tuple()
            for _ in range(number):
                layers = layers + (32,)
            return layers
        except Exception as e:
            print("[ERR] The following error occured while trying to get the total layers: "+str(e))

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

            layers = self.get_layers(num_layers)
            
            print("[MLP] Creating a new instance of a Multi Layer Perceptron!")
            self.__model = MLPClassifier(hidden_layer_sizes=layers, max_iter=1000, activation='relu', solver='adam', random_state=42)
            print("[MLP] A new instance for Multi Layer Perceptron has been created with the following hyper params: ")
            print("[MLP] Size of Hidden Layers: "+str(num_layers))
            print("[MLP] Max Iterations: 1000")
            print("[MLP] Activation Function: ReLu ( x if x > 0 else 0 )")
            print("[MLP] Solver: Adam.")

            print("[MLP] Training this new instance of MLP, please wait this might take more than a minute.")
            self.__model.fit(self.__X_trian, self.__y_train)
            print("[MLP] Object reference for the trained and newly fitted MLP model: "+str(self.__model))
            self.get_model_metrics() #get the classification metrics of the model.
            return self.__model
        except Exception as e:
            print("[ERR] The following error occured while trying to train an MLP classifier: "+str(e))

    def get_actuals(self, y_pred) -> list:
        """
        Reads the actual labels for the test set and then generates the y_test labeled set.
        """
        try:
            y_test = pd.read_csv("./datasets/test_labels.csv")
            y_test = y_test.drop(columns=['id', 'severe_toxic', 'obscene','threat','insult','identity_hate'])
            print(y_test)
            y_test = y_test['toxic'].replace(-1, 1)
            y_test = y_test.values
            print(classification_report(y_test, y_pred))
        except Exception as e:
            print("[ERR] the following error occured while trying to get the metrics of test: "+str(e))

    def make_prediction(self, probabilities, threshold = 0.3):
        """
        Makes a prediction based on the probabilities of each comment being toxic, and non toxic.
        """
        try:
            results = list()
            for probability in probabilities:
                if probability[1] >= threshold:
                    results.append(1)
                else:
                    results.append(0)

            return results
        except Exception as e:
            print("[ERR] The following error occured while trying to make predictions: "+str(e))
    
    def test_MLP_model(self, path_to_test_file: str, MLP_model: MLPClassifier) -> None:
        """
        Tests the currently passed model against a test dataset, and then prints it's metrics as well.
        This method will also create a output.csv which will be unique each time.
        """
        try:
            if MLP_model is None:
                raise Exception("None Object passed as a parameter instead of MLPClassifier!")
            elif self.__word2vec_model is None:
                raise Exception("No Word2Vec model is has been trained please run the train method before running the test method!")
            else:
                print("[TEST] Testing the model now!")
                self.read_data(path_to_test_file) #read the test file.
                self.get_five() #print the resulting set
                self.clean_data() #create a new feature
                self.create_embeddings() # create embeddings based on this new feature
                self.get_five()
                self.__X_test = np.stack(self.__data_frame['embedding'].values)
                y_pred = self.__model.predict_proba(self.__X_test)
                
                not_toxic_probabilities = list()
                toxic_probabilities = list()

                for probabilities in y_pred:
                    not_toxic_probabilities.append(probabilities[0])
                    toxic_probabilities.append(probabilities[1])
                
                self.__data_frame['not_toxic_proba'] = not_toxic_probabilities
                self.__data_frame['toxic_proba'] = toxic_probabilities 
                self.__data_frame['toxic_label'] = self.make_prediction(probabilities=y_pred)
                self.get_five()
                print("[TEST] writing output to output.csv")
                self.__data_frame.to_csv("output.csv")
        except Exception as e:
            print("[ERR] The following error occured while trying to test the method: "+str(e))

