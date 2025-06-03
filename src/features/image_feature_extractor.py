import os

import cv2
import dlib
from skimage.feature import hog
import numpy as np

from .base_extractor import BaseFeatureExtractor
from src.utils.utils import *


class ImageFeatureExtractor(BaseFeatureExtractor):

    def __init__(self):
        """
        predictor_path: ścieżka do shape_predictor_68_face_landmarks.dat
        """
        self.detector = dlib.get_frontal_face_detector()

    def extract_features(self, image_files):
        """Ekstrahuje cechy z obrazów — face crop z paddingiem i resize do 96x96"""
        features = []
        filenames = []

        for img_path in image_files:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            rects = self.detector(image, 0)
            if len(rects) == 0:
                continue

            rect = rects[0]
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

            face_crop = image[y1:y2, x1:x2]

            h, w = face_crop.shape
            size = max(h, w)
            square = np.zeros((size, size), dtype=face_crop.dtype)
            square[0:h, 0:w] = face_crop

            face_resized = cv2.resize(square, (96, 96))

            face_normalized = face_resized.astype(np.float32) / 255.0

            features.append(face_normalized.flatten())
            filenames.append(os.path.basename(img_path))

        return features, filenames






