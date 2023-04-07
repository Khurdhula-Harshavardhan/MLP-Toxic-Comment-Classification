import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


class MLP_Classifier():
    __model = None
    def __init__(self) -> None:
        pass

    def train_MLP_model(path_to_train_file: str, num_layers = 2) -> MLPClassifier:
        """
        This method, trains a Multi-layer preceptron and then returns it.
        This training is performed using Embeddings on comment_text of user.
        """
        try:
            pass
        except Exception as e:
            print("[ERR] The following error occured while trying to train an MLP classifier: "+str(e))