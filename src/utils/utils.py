import os

import pickle


def extract_label_from_filename(filename):
    """Ekstrahuje etykietę z nazwy pliku obrazka, np. test_angry_0001.jpg"""
    base = os.path.basename(filename)
    label = base.split('-')[0]

    return label


def save_features_to_pickle(filename, data):
    """Zapisuje cechy do pliku pickle"""
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_features_from_pickle(filename):
    """Czyta cechy z pliku pickle"""
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def save_history(history, model_name, history_dir='models/history'):
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, f'{model_name}_history.pickle')

    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
