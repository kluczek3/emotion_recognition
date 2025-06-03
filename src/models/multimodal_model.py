from keras import Input
from keras.src.layers import Dropout, Dense, Concatenate
from keras.src.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

from .base_model import BaseModel
from src.config import *
from src.utils.utils import *

from keras.models import load_model, Model


class MultimodalModel(BaseModel):
    def __init__(self, image_model_path='models/saved/model_images.keras', audio_model_path='models/saved/model_audio.keras', num_classes=None):
        self.image_model_path = image_model_path
        self.audio_model_path = audio_model_path

        self.X_audio, self.y = self.load_dataset_from_pickle(os.path.join(DATA_DIR, 'processed', 'train-audio-features.pickle'))
        self.X_images = self.load_and_filter_image_data(os.path.join(DATA_DIR, 'processed', 'train-image-features.pickle'))

        self.X_audio_test, self.y_test = self.load_dataset_from_pickle(os.path.join(DATA_DIR, 'processed', 'test-audio-features.pickle'))
        self.X_images_test = self.load_and_filter_image_data(os.path.join(DATA_DIR, 'processed', 'test-image-features.pickle'))
        self.X_images = self.X_images.reshape(-1, 96, 96, 1)
        self.X_images_test = self.X_images_test.reshape(-1, 96, 96, 1)

        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)
        self.y_test = self.label_encoder.fit_transform(self.y_test)

        self.input_shape_image = self.X_images.shape[1:]
        self.input_shape_audio = self.X_audio.shape[1:]
        self.num_classes = len(np.unique(self.y)) if num_classes is None else num_classes
        self.model = None

    def build_model(self, freeze_encoders=True):
        image_model = load_model(self.image_model_path)
        audio_model = load_model(self.audio_model_path)

        image_model._name = 'image_encoder'
        audio_model._name = 'audio_encoder'

        image_model.trainable = not freeze_encoders
        audio_model.trainable = not freeze_encoders

        for layer in image_model.layers:
            layer._name = "image_" + layer.name

        for layer in audio_model.layers:
            layer._name = "audio_" + layer.name

        img_input = Input(shape=image_model.input_shape[1:], name='img_input')
        audio_input = Input(shape=audio_model.input_shape[1:], name='audio_input')

        img_output = image_model(img_input)
        audio_output = audio_model(audio_input)

        fused = Concatenate(name='fusion_concat')([img_output, audio_output])

        x = Dense(256, activation='relu', name='fusion_dense_1')(fused)
        x = Dropout(0.5, name='fusion_dropout_1')(x)
        x = Dense(128, activation='relu', name='fusion_dense_2')(x)
        x = Dropout(0.5, name='fusion_dropout_2')(x)
        output = Dense(self.num_classes, activation='softmax', name='fusion_output')(x)

        self.model = Model(inputs=[img_input, audio_input], outputs=output, name='multimodal_model')

        self.model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self):
        history = self.model.fit([self.X_images, self.X_audio], self.y, epochs=10, batch_size=32, validation_data=None)
        save_history(history, 'multimodal')
        return history

    def evaluate(self):
        return self.model.evaluate([self.X_images_test, self.X_audio_test], self.y_test)

    def predict(self):
        return self.model.predict([self.X_images_test, self.X_audio_test])

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path, image_model_path='models/saved/model_images.keras',
                   audio_model_path='models/saved/model_audio.keras'):
        self.model = load_model(path)

    @staticmethod
    def load_and_filter_image_data(pickle_path, suffix='-02.jpg'):
        data = load_features_from_pickle(pickle_path)
        X = []
        y = []
        for fname, features, label in data:
            if fname.endswith(suffix):
                X.append(features)
                y.append(label)

        return np.array(X)
