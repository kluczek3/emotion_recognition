import os

import cv2
import dlib
from skimage.feature import hog
import numpy as np

from .base_extractor import BaseFeatureExtractor
from src.utils.utils import *


class ImageFeatureExtractor(BaseFeatureExtractor):

    def __init__(self, predictor_path):
        """
        predictor_path: ścieżka do shape_predictor_68_face_landmarks.dat
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def extract_features(self, image_files):
        """Ekstrahuje cechy z obrazów za pomocą landmarków dlib"""
        features = []
        filenames = []

        for img_path in image_files:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            rects = self.detector(image, 1)
            if len(rects) == 0:
                continue

            rect = rects[0]
            shape = self.predictor(image, rect)

            landmarks = np.array([[p.x, p.y] for p in shape.parts()])

            landmarks = landmarks.astype(np.float32)
            landmarks[:, 0] /= image.shape[1]
            landmarks[:, 1] /= image.shape[0]

            features.append(landmarks.flatten())
            filenames.append(os.path.basename(img_path))

            break

        return features, filenames






