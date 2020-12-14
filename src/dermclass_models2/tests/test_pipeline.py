import pytest
import pandas as pd
import tensorflow as tf

from dermclass_models2.pipeline import StructuredPipeline, TextPipeline, ImagePipeline


class Test_SklearnPipeline:

    def test_fit_structured_data(self):
        pass

    def test__get_sklearn_model(self):
        pass

    @staticmethod
    def test__hyper_param_optimization():
        pass

    def test__set_dfs(self):
        pass

    def test__set_dfs_test(self):
        pass

    def test__tune_hyperparameters(self):
        pass


class Test_TfPipeline:

    def test_fit_datasets(self):
        pass

    def test__compile_model(self):
        pass

    def test__train_model(self):
        pass


class TestStructuredPipeline:

    def test_get_processing_pipeline(self):
        pass

    def test_get_model(self):
        pass

    def test_get_modeling_pipeline(self):
        pass


class TestImagePipeline:

    def test_set_img_size_and_model_obj(self):
        pass

    def test_get_processing_pipeline(self):
        pass

    def test_get_model(self):
        pass

    def test_get_modeling_pipeline(self):
        pass


class TestTextPipeline:

    @staticmethod
    def test_encode_dataset():
        pass

    def test_get_best_modeling_pipeline_type(self):
        pass

    def test_get_model(self):
        pass

    def test_get_processing_pipeline(self):
        pass

    def test_get_modeling_pipeline(self):
        pass
