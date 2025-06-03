from keras import Input
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Conv1D, GlobalAveragePooling1D, Activation, MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.src.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.src.optimizers import Adam

from .base_model import BaseModel
from src.config import *
from src.utils.utils import *


class AudioRNNModel(BaseModel):
    def __init__(self, input_shape=None, num_classes=None):
        self.model = None
        self.X, self.y = self.load_dataset_from_pickle(os.path.join(DATA_DIR, 'processed', 'train-audio-features.pickle'))
        self.X_test, self.y_test = self.load_dataset_from_pickle(os.path.join(DATA_DIR, 'processed', 'test-audio-features.pickle'))

        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)
        self.y_test = self.label_encoder.fit_transform(self.y_test)

        self.input_shape = self.X.shape[1:] if input_shape is None else input_shape
        self.num_classes = len(np.unique(self.y)) if num_classes is None else num_classes

    def build_model(self):
        if self.input_shape is None or self.num_classes is None:
            raise ValueError("input_shape and num_classes must be set before building the model")

        model = Sequential([
            Input(shape=self.input_shape),

            Conv1D(filters=32, kernel_size=9),
            BatchNormalization(),
            Activation('elu'),
            MaxPooling1D(pool_size=2, strides=2),
            Dropout(0.25),

            Conv1D(filters=64, kernel_size=7),
            BatchNormalization(),
            Activation('elu'),
            MaxPooling1D(pool_size=2, strides=2),
            Dropout(0.25),

            Conv1D(filters=128, kernel_size=5),
            BatchNormalization(),
            Activation('elu'),
            MaxPooling1D(pool_size=2, strides=2),
            Dropout(0.25),

            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),

            GlobalAveragePooling1D(),

            Dense(units=self.num_classes, activation='softmax')
        ], name='audio_model')

        model.compile(optimizer=Adam(learning_rate=0.0005),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=60, batch_size=16):
        if self.model is None:
            self.build_model()

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        opt = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[opt, early_stop]
        )
        save_history(history, 'audio')
        return history

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not built or loaded")
        preds = self.model.predict(X)
        return preds.argmax(axis=1)

    def evaluate(self):
        if self.model is None:
            raise ValueError("Model is not built or loaded")
        return self.model.evaluate(self.X_test, self.y_test, verbose=0)

    def save_model(self, save_path):
        if self.model is None:
            raise ValueError("Model is not built or loaded")
        self.model.save(save_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)
