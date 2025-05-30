import os
from glob import glob

import librosa.feature
import numpy as np

from .base_extractor import BaseFeatureExtractor
from src.utils.utils import *


class AudioFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, sampling_rate=44100):
        self.sr = sampling_rate

    def extract_features(self, image_files):
        """Ekstrahuje cechy z plików audio"""
        features = []
        filenames = []

        for audio_path in image_files:
            mfcc = self.extract_mfcc(audio_path)
            chroma = self.extract_chroma(audio_path)
            sc = self.extract_spectral_contrast(audio_path)

            vector = np.hstack([mfcc, chroma, sc])
            features.append(vector)
            filenames.append(os.path.basename(audio_path))

            break

        return features, filenames

    def extract_mfcc(self, audio_path: str):
        """Ekstrahuje MFCC"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        return [np.mean(mfcc), np.std(mfcc)]

    def extract_chroma(self, audio_path: str):
        """Ekstrahuje chroma"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        chroma = librosa.feature.chroma_stft(y=y)

        return [np.mean(chroma), np.std(chroma)]

    def extract_spectral_contrast(self, audio_path: str):
        """Ekstrahuje spectral contrast"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)

        return [np.mean(sc), np.std(sc)]

