class BaseFeatureExtractor:
    def extract_features(self, data_path: str):
        """Ekstrahuje cechy z podanej ścieżki danych"""
        raise NotImplementedError
