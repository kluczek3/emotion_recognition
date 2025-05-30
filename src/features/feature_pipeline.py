import os
from glob import glob

import numpy as np

from .audio_feature_extractor import AudioFeatureExtractor
from .image_feature_extractor import ImageFeatureExtractor
from src.config import *
from src.models.model_factory import ModelFactory


class FeaturePipeline:
    def __init__(self):
        self.afe = AudioFeatureExtractor()
        self.ife = ImageFeatureExtractor('models/shape_predictor_68_face_landmarks.dat')

    @staticmethod
    def load_data(path, ext):
        """Wczytuje dostępne dane o zadanym rozszerzeniu"""

        return glob(os.path.join(BASE_DIR, path, '**', f'*.{ext}'), recursive=True)

    def extract_image_features(self, split='train'):
        data_path = os.path.join('data', 'processed', 'images', split)
        images_path = self.load_data(data_path, 'jpg')
        features, image_names = self.ife.extract_features(images_path)

        return features, image_names

    def extract_audio_features(self, split='train'):
        """Wykrywa cechy audio przy użyciu extractora"""
        data_path = os.path.join('data', 'processed', 'audio', split)
        images_path = self.load_data(data_path, 'wav')
        features, audio_names = self.afe.extract_features(images_path)

        return features, audio_names

    def combine_features(self, image_features, audio_features):
        """Łączy cechy obrazu i audio""" # może nie być w użyciu
        return np.hstack([image_features, audio_features])

