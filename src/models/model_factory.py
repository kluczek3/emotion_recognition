from .cnn_model import CNNModel
from .audio_rnn_model import AudioRNNModel
from .multimodal_model import MultimodalModel


class ModelFactory:
    @staticmethod
    def get_model(model_type: str):
        if model_type == 'cnn':
            return CNNModel()
        elif model_type == 'audio_rnn':
            return AudioRNNModel()
        elif model_type == 'multimodal':
            return MultimodalModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
