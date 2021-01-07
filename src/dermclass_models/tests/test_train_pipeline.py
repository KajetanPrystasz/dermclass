import pytest
import copy
import tensorflow as tf

from dermclass_models.train_pipeline import (StructuredTrainPipeline, TextTrainPipeline,
                                             ImageTrainPipeline, run_controller)
from dermclass_models.preprocessing import StructuredPreprocessor, TextPreprocessors, ImagePreprocessors
from dermclass_models.pipeline import StructuredPipeline, TextPipeline, ImagePipeline

_validation_dict = {"NUMERIC_NA_NOT_ALLOWED": ["age"],
                    "ORDINAL_NA_NOT_ALLOWED": ["erythema"],
                    "CATEGORICAL_NA_NOT_ALLOWED": [],
                    "NUMERIC_NA_ALLOWED": [],
                    "ORDINAL_NA_ALLOWED": [],
                    "CATEGORICAL_NA_ALLOWED": []}


def test_run_controller(monkeypatch):
    with pytest.raises(RuntimeError):
        run_controller("not_existing_pipeline_type")

    monkeypatch.setattr(StructuredTrainPipeline, "run", lambda x: None)
    monkeypatch.setattr(TextTrainPipeline, "run", lambda x: None)
    monkeypatch.setattr(ImageTrainPipeline, "run", lambda x: None)

    run_controller()


class TestStructuredTrainPipeline:

    @pytest.mark.pipeline_training
    def test_run(self, testing_config, monkeypatch, tmp_path, structured_set, structured_training_df, xgboost_trial):
        testing_config = copy.copy(testing_config)
        testing_config.DEFAULT_BEST_MODEL = "XGBClassifier"
        testing_config.TRIALS_DICT = {"XGBClassifier": xgboost_trial}
        testing_config.NA_VALIDATION_VAR_DICT = _validation_dict

        testing_config.PICKLE_DIR = tmp_path / "pickle_dir"
        testing_config.PICKLE_DIR.mkdir()
        testing_config.PIPELINE_TYPE = "structured_pipeline"

        train_pipeline = StructuredTrainPipeline(testing_config)

        train_pipeline.preprocessor = StructuredPreprocessor(testing_config)
        train_pipeline.pipeline = StructuredPipeline(testing_config)

        def mock_load_data(path=None):
            return structured_set

        monkeypatch.setattr(train_pipeline.preprocessor, "load_data", mock_load_data)
        train_pipeline.run()

        assert [x for x in testing_config.PICKLE_DIR.iterdir()][0].exists()


class TestTextTrainPipeline:

    @pytest.mark.pipeline_training
    def test_run(self, testing_config, monkeypatch, tmp_path,
                 text_train_dataset, structured_text_set, multinomial_nb_trial):
        testing_config = copy.copy(testing_config)
        testing_config.TRIALS_DICT = {"MultinomialNB": multinomial_nb_trial}
        testing_config.DEFAULT_BEST_MODEL = "MultinomialNB"
        testing_config.PICKLE_DIR = tmp_path / "pickle_dir"
        testing_config.PICKLE_DIR.mkdir()
        testing_config.PIPELINE_TYPE = "text_pipeline"

        train_pipeline = TextTrainPipeline(testing_config)

        train_pipeline.preprocessor = TextPreprocessors(testing_config)
        train_pipeline.pipeline = TextPipeline(testing_config)

        def mock_load_data(get_datasets, path=None):
            if get_datasets:
                return text_train_dataset, text_train_dataset, text_train_dataset
            else:
                x_train, x_test, y_train, y_test = structured_text_set
                return x_train, x_test, y_train, y_test

        monkeypatch.setattr(train_pipeline.preprocessor, "load_data", mock_load_data)
        train_pipeline.run()

        assert [x for x in testing_config.PICKLE_DIR.iterdir()][0].exists()


class TestImageTrainPipeline:

    @pytest.mark.pipeline_training
    def test_run(self, testing_config, monkeypatch, tmp_path, image_train_dataset):
        train_pipeline = ImageTrainPipeline(testing_config)
        testing_config.LEARNING_RATE = 0.01
        testing_config.PICKLE_DIR = tmp_path / "pickle_dir"
        testing_config.PICKLE_DIR.mkdir()
        testing_config.PIPELINE_TYPE = "text_pipeline"

        train_pipeline.preprocessor = ImagePreprocessors(testing_config)
        train_pipeline.pipeline = ImagePipeline(testing_config)

        def mock_load_data(path=None):
            img_size = (456, 456)
            model = tf.keras.applications.EfficientNetB5

            train_pipeline.preprocessor.img_size = img_size
            train_pipeline.preprocessor.model = model

            return image_train_dataset, image_train_dataset, image_train_dataset

        monkeypatch.setattr(train_pipeline.preprocessor, "load_data", mock_load_data)

        train_pipeline.run()

        assert [x for x in testing_config.PICKLE_DIR.iterdir()][0].exists()
