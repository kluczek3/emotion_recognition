import numpy as np

from src.utils.utils import *


class BaseModel:
    def build_model(self):
        """Buduje model - do implementacji w klasie potomnej."""
        raise NotImplementedError

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Trenuje model - do implementacji w klasie potomnej."""
        raise NotImplementedError

    def predict(self, X):
        """Predykcja na podstawie danych X - do implementacji w klasie potomnej."""
        raise NotImplementedError

    def evaluate(self, X_test, y_test):
        """Ocena modelu - do implementacji w klasie potomnej."""
        raise NotImplementedError

    def save_model(self, save_path):
        """Zapisuje model - do implementacji w klasie potomnej."""
        raise NotImplementedError

    def load_model(self, model_path):
        """Ładuje model - do implementacji w klasie potomnej."""
        raise NotImplementedError

    @staticmethod
    def load_dataset_from_pickle(filepath):
        data = load_features_from_pickle(filepath)
        X = []
        y = []
        names = []
        for fname, features, label in data:
            names.append(fname)
            X.append(features)
            y.append(label)
        return np.array(X), np.array(y)
