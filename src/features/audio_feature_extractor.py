import os
from glob import glob

import librosa
import librosa.feature
import librosa.effects
from sklearn.preprocessing import StandardScaler

from .base_extractor import BaseFeatureExtractor
from src.utils.utils import *
from keras.preprocessing.sequence import pad_sequences


class AudioFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, sampling_rate=16000):
        self.sr = sampling_rate

    def extract_features(self, audio_files, augment=True):
        """Ekstrahuje cechy z plików audio wraz z opcjonalną augmentacją danych"""
        features = []
        filenames = []

        for audio_path in audio_files:
            y, sr = librosa.load(audio_path, sr=self.sr)

            y_versions = []

            y_versions.append((os.path.basename(audio_path), y))

            if augment:
                noise_amp = 0.005 * np.random.uniform() * np.amax(y)
                y_noise = y + noise_amp * np.random.normal(size=y.shape[0])
                y_versions.append((os.path.basename(audio_path) + '_noise', y_noise))

                y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
                y_versions.append((os.path.basename(audio_path) + '_pitch_up', y_pitch_up))

                y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
                y_versions.append((os.path.basename(audio_path) + '_pitch_down', y_pitch_down))

            for variant_name, y_var in y_versions:
                mfcc = self.extract_mfcc(y_var, sr)
                mel = self.extract_mel_spectrogram(y_var, sr)
                chroma = self.extract_chroma(y_var, sr)
                zcr = self.extract_zero_crossing_rate(y_var)
                sc = self.extract_spectral_contrast(y_var, sr)
                tonn = self.extract_tonnetz(y_var, sr)

                vector = np.hstack([mfcc, mel, zcr, chroma, sc, tonn])
                features.append(vector)
                filenames.append(variant_name)

        all_frames = np.vstack(features)
        scaler = StandardScaler()
        scaler.fit(all_frames)

        scaled_features = [scaler.transform(f) for f in features]

        padded_features = pad_sequences(
            scaled_features,
            maxlen=400,
            dtype='float32',
            padding='post',
            truncating='post'
        )

        return padded_features, filenames
    def extract_mfcc(self, y, sr):
        """Ekstrahuje MFCC (i jego pochodne) z zadanego sygnału"""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30, n_fft=512, hop_length=160)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([mfcc, delta, delta2])
        return combined.T
    def extract_chroma(self, y, sr):
        """Ekstrahuje chroma z zadanego sygnału"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=512, hop_length=160)
        return chroma.T

    def extract_spectral_contrast(self, y, sr):
        """Ekstrahuje spectral contrast z zadanego sygnału"""
        sc = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=512, hop_length=160)
        return sc.T

    def extract_mel_spectrogram(self, y, sr):
        """Ekstrahuje mel-spektrogram (z konwersją do dB) z zadanego sygnału"""
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=512, hop_length=160)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.T

    def extract_zero_crossing_rate(self, y):
        """Ekstrahuje zero crossing rate z zadanego sygnału"""
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=160)
        return zcr.T

    def extract_tonnetz(self, y, sr):
        """Ekstrahuje cechy harmoniczne Tonnetz z sygnału"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=512, hop_length=160)
        tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)
        return tonnetz.T
