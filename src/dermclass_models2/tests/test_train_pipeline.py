import pytest

from dermclass_models2.train_pipeline import (StructuredTrainPipeline, TextTrainPipeline,
                                              ImageTrainPipeline, run_controller)
from dermclass_models2.preprocessing import StructuredPreprocessor, TextPreprocessors, ImagePreprocessors
from dermclass_models2.pipeline import StructuredPipeline, TextPipeline, ImagePipeline


def test_run_controller(monkeypatch):
    with pytest.raises(RuntimeError):
        run_controller("not_existing_pipeline_type")

    monkeypatch.setattr(StructuredTrainPipeline, "run", lambda x: None)
    monkeypatch.setattr(TextTrainPipeline, "run", lambda x: None)
    monkeypatch.setattr(ImageTrainPipeline, "run", lambda x: None)

    run_controller()


# TODO: Finish slow tests...
# class TestStructuredTrainPipeline:
#
#     @pytest.mark.slow
#     def test_run(self, testing_config, monkeypatch, structured_set):
#         train_pipeline = StructuredTrainPipeline(testing_config)
#
#         train_pipeline.preprocessor = StructuredPreprocessor(testing_config)
#         train_pipeline.pipeline = StructuredPipeline(testing_config)
#
#         monkeypatch.setattr(train_pipeline.preprocessor, "load_data", lambda path: structured_set)
#         monkeypatch.setattr(train_pipeline, "_save_modeling_pipeline", lambda modeling_pipeline, backend: None)
#         train_pipeline.run()
#
#         assert True
#
#
# class TestTextTrainPipeline:
#
#     @pytest.mark.slow
#     def test_run(self, testing_config, monkeypatch, text_train_dataset, structured_text_set):
#         train_pipeline = TextTrainPipeline(testing_config)
#
#         train_pipeline.preprocessor = TextPreprocessors(testing_config)
#         train_pipeline.pipeline = TextPipeline(testing_config)
#
#         def mock_load_data(get_datasets, path=None):
#             if get_datasets:
#                 return text_train_dataset, text_train_dataset, text_train_dataset
#             else:
#                 x_train, x_test, y_train, y_test = structured_text_set
#                 return x_train, x_test, y_train, y_test
#
#         monkeypatch.setattr(train_pipeline.preprocessor, "load_data", mock_load_data)
#         monkeypatch.setattr(train_pipeline, "_save_modeling_pipeline", lambda modeling_pipeline, backend: None)
#         train_pipeline.run()
#
#         assert True
#
#
# class TestImageTrainPipeline:
#
#     @pytest.mark.slow
#     def test_run(self, testing_config, monkeypatch, image_train_dataset):
#         train_pipeline = ImageTrainPipeline(testing_config)
#
#         train_pipeline.preprocessor = ImagePreprocessors(testing_config)
#         train_pipeline.pipeline = ImagePipeline(testing_config)
#
#         def mock_load_data(path=None):
#             return image_train_dataset, image_train_dataset, image_train_dataset
#
#         monkeypatch.setattr(train_pipeline.preprocessor, "load_data", mock_load_data)
#         monkeypatch.setattr(train_pipeline, "_save_modeling_pipeline", lambda modeling_pipeline, backend: None)
#         train_pipeline.run()
#
#         assert True
