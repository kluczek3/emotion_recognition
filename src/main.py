from features.feature_pipeline import FeaturePipeline
from data.dataset_builder import DataSetBuilder
from models.model_factory import ModelFactory


def main():
    """
    fp = FeaturePipeline()
    dsb = DataSetBuilder(fp)
    dsb.build_dataset('test', False)
    dsb.build_dataset('train', False)
    """
    mf = ModelFactory()
    model = mf.get_model('multimodal')

    model.build_model()
    model.train()
    model.save_model('models/saved/model_multimodal.keras')

    test_loss, test_acc = model.evaluate()
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    main()
