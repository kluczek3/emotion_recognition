from keras import Input
from keras.src.layers import Reshape
from sklearn.preprocessing import LabelEncoder

from .base_model import BaseModel
from src.config import *


from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from src.utils.utils import *


class CNNModel(BaseModel):
    def __init__(self, input_shape=(96, 96, 1), num_classes=None):
        self.model = None
        self.X, self.y = self.load_dataset_from_pickle(os.path.join(DATA_DIR, 'processed', 'train-image-features.pickle'))
        self.X_test, self.y_test = self.load_dataset_from_pickle(os.path.join(DATA_DIR, 'processed', 'test-image-features.pickle'))

        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)
        self.X = self.X.reshape(-1, 96, 96, 1)

        self.X_test = self.X_test.reshape(-1, 96, 96, 1)
        self.y_test = self.label_encoder.fit_transform(self.y_test)

        self.input_shape = input_shape
        self.num_classes = len(np.unique(self.y)) if num_classes is None else num_classes

    def build_model(self):
        if self.input_shape is None or self.num_classes is None:
            raise ValueError("input_shape and num_classes must be set before building the model")

        model = Sequential(name='images_model')
        model.add(Input(shape=self.input_shape))
        model.add(Reshape((self.input_shape[0], self.input_shape[1], 1)))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=0.0005),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=64):

        if self.model is None:
            self.build_model()

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stop],
        )
        save_history(history, 'images')
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
