import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_IMAGES_DIR = os.path.join(DATA_DIR, 'processed', 'images')
PROCESSED_AUDIO_DIR = os.path.join(DATA_DIR, 'processed', 'audio')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
