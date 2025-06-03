from collections import defaultdict

from src.utils.utils import *
from src.config import *


class DataSetBuilder:
    def __init__(self, feature_pipeline):
        self.feature_pipeline = feature_pipeline

    def build_dataset(self, split='train', augment=True):
        image_features, image_names = self.feature_pipeline.extract_image_features(split)
        audio_features, audio_names = self.feature_pipeline.extract_audio_features(split, augment)

        filename = os.path.join(BASE_DIR, 'data', 'processed', f'{split}-audio-features.pickle')
        audio_data = [(fname, feat, extract_label_from_filename(fname)) for fname, feat in zip(audio_names, audio_features)]
        save_features_to_pickle(filename, audio_data)

        filename = os.path.join(DATA_DIR, 'processed', f'{split}-image-features.pickle')
        image_data = [(fname, feat, extract_label_from_filename(fname)) for fname, feat in zip(image_names, image_features)]
        save_features_to_pickle(filename, image_data)




