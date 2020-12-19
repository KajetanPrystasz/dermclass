import copy

import pytest
import pandas as pd
import numpy as np
import tensorflow as tf

from dermclass_models2.pipeline import StructuredPipeline, TextPipeline, ImagePipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator

from transformers import TFDistilBertForSequenceClassification

_validation_dict = {"NUMERIC_NA_NOT_ALLOWED": ["age"],
                    "ORDINAL_NA_NOT_ALLOWED": ["erythema"],
                    "CATEGORICAL_NA_NOT_ALLOWED": [],
                    "NUMERIC_NA_ALLOWED": [],
                    "ORDINAL_NA_ALLOWED": [],
                    "CATEGORICAL_NA_ALLOWED": []}


def _assert_structured_data(pipeline, x_train, x_test, y_train, y_test):
    assert pipeline.x_train.equals(x_train)
    assert pipeline.x_test.equals(x_test)
    assert pipeline.y_train.equals(y_train)
    assert pipeline.y_test.equals(y_test)


def _xgboost_trial(trial) -> dict:
    params = {"subsample": trial.suggest_discrete_uniform("subsample", 0.1, 1, 0.1),
              "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.6, 1, 0.1)}
    return params


class TestStructuredPipeline:

    def test_get_processing_pipeline(self, testing_config):
        testing_config.NA_VALIDATION_VAR_DICT = _validation_dict

        pipeline = StructuredPipeline(testing_config)

        processing_pipeline = pipeline.get_processing_pipeline()

        assert isinstance(processing_pipeline, (Pipeline, ColumnTransformer))

    def test_get_model(self, testing_config, structured_set):

        testing_config.TUNING_FUNC_PARAMS = {"n_jobs": -1, "max_overfit": 0.9, "cv": 2, "n_trials": 1}
        testing_config.DEFAULT_BEST_MODEL = "XGBClassifier"
        testing_config.TRIALS_DICT = {"XGBClassifier": _xgboost_trial}
        testing_config.SEED = 42
        testing_config.NA_VALIDATION_VAR_DICT = _validation_dict

        x_train, x_test, y_train, y_test = structured_set

        pipeline = StructuredPipeline(testing_config)
        pipeline.fit_structured_data(x_train, x_test, y_train, y_test)

        pipeline._hyper_param_optimization = (lambda trial, model_name, trial_func, max_overfit, cv,
                                                     x_train, x_test, y_train, y_test: 1)

        model = pipeline.get_model()

        assert isinstance(model, BaseEstimator)

    def test_get_modeling_pipeline(self, testing_config, structured_set):

        testing_config.TUNING_FUNC_PARAMS = {"n_jobs": -1, "max_overfit": 0.9, "cv": 2, "n_trials": 1}
        testing_config.TRIALS_DICT = {"XGBClassifier": _xgboost_trial}

        testing_config.SEED = 42
        testing_config.NA_VALIDATION_VAR_DICT = _validation_dict

        x_train, x_test, y_train, y_test = structured_set

        pipeline = StructuredPipeline(testing_config)
        pipeline.fit_structured_data(x_train, x_test, y_train, y_test)

        pipeline._hyper_param_optimization = (lambda trial, model_name, trial_func, max_overfit, cv,
                                                     x_train, x_test, y_train, y_test: 1)
        modeling_pipeline = pipeline.get_modeling_pipeline(x_train, x_test, y_train, y_test)

        assert isinstance(modeling_pipeline, Pipeline)
        assert isinstance(modeling_pipeline.steps[0][1], (Pipeline, ColumnTransformer))
        assert isinstance(modeling_pipeline.steps[1][1], (BaseEstimator))

    def test_fit_structured_data(self, testing_config, structured_set):
        testing_config.TUNING_FUNC_PARAMS = {"n_jobs": -1, "max_overfit": 0.9, "cv": 2, "n_trials": 1}
        pipeline = StructuredPipeline(testing_config)

        x_train, x_test, y_train, y_test = structured_set
        pipeline.fit_structured_data(x_train, x_test, y_train, y_test)

        _assert_structured_data(pipeline, x_train, x_test, y_train, y_test)


class TestImagePipeline:

    def test_fit_datasets(self, testing_config, train_dataset):
        testing_config.PATIENCE = 3
        pipeline = ImagePipeline(testing_config)
        pipeline.fit_datasets(train_dataset, train_dataset, train_dataset)

        assert pipeline.train_dataset == train_dataset
        assert pipeline.validation_dataset == train_dataset
        assert pipeline.test_dataset == train_dataset

    def test_set_img_size_and_model_obj(self, testing_config, train_dataset):
        testing_config.PATIENCE = 3
        pipeline = ImagePipeline(testing_config)
        pipeline.set_img_size_and_model_obj((456, 456), tf.keras.applications.EfficientNetB5)

        assert pipeline.img_size == (456, 456)
        assert pipeline.model_obj == tf.keras.applications.EfficientNetB5

    def test_get_processing_pipeline(self, testing_config):
        testing_config.PATIENCE = 3
        pipeline = ImagePipeline(testing_config)

        pipeline_rescale = pipeline.get_processing_pipeline(rescale=True)
        pipeline_no_rescale = pipeline.get_processing_pipeline(rescale=False)

        assert isinstance(pipeline_rescale, tf.keras.Sequential)
        assert isinstance(pipeline_no_rescale, tf.keras.Sequential)
        assert isinstance(pipeline_rescale.get_layer(index=-1),
                          tf.keras.layers.experimental.preprocessing.Rescaling)

    def test_get_model(self, testing_config):
        testing_config.PATIENCE = 3
        testing_config.DISEASES = ["test_disease1", "test_disease2", "test_disease3"]
        pipeline = ImagePipeline(testing_config)

        model = pipeline.get_model(tf.keras.applications.EfficientNetB5)

        assert model.trainable is False

    def test_get_modeling_pipeline(self, testing_config, image_train_dataset):
        testing_config.PATIENCE = 3
        testing_config.DISEASES = ["test_disease1", "test_disease2", "test_disease3"]
        testing_config.img_size = (456, 456)
        testing_config.LEARNING_RATE = 0.01
        testing_config.METRICS = "accuracy"
        testing_config.NUM_EPOCHS = 1

        pipeline = ImagePipeline(testing_config)
        pipeline.fit_datasets(image_train_dataset, image_train_dataset, image_train_dataset)
        pipeline.set_img_size_and_model_obj(testing_config.img_size,
                                            tf.keras.applications.EfficientNetB5)

        modeling_pipeline = pipeline.get_modeling_pipeline()


class TestTextPipeline:

    def test_fit_structured_data(self, testing_config, structured_set):
        testing_config.PATIENCE = 3
        pipeline = TextPipeline(testing_config)

        x_train, x_test, y_train, y_test = structured_set
        pipeline.fit_structured_data(x_train, x_test, y_train, y_test)

        _assert_structured_data(pipeline, x_train, x_test, y_train, y_test)

    def test_fit_datasets(self, testing_config, train_dataset):
        testing_config.PATIENCE = 3
        pipeline = TextPipeline(testing_config)
        pipeline.fit_datasets(train_dataset, train_dataset, train_dataset)

        assert pipeline.train_dataset == train_dataset
        assert pipeline.validation_dataset == train_dataset
        assert pipeline.test_dataset == train_dataset

    @staticmethod
    def test_encode_dataset():
        pass

    def test_get_best_modeling_pipeline_type(self, testing_config, structured_set, train_dataset, monkeypatch):
        pipeline = TextPipeline(testing_config)

        x_train, x_test, y_train, y_test = structured_set

        pipeline.fit_structured_data(x_train, x_test, y_train, y_test)
        pipeline.fit_datasets(train_dataset, train_dataset, train_dataset)

        sklearn_pipeline = pipeline.get_modeling_pipeline(use_sklearn=True)
        tf_pipeline = pipeline.get_modeling_pipeline(use_sklearn=False,
                                                     train_dataset=train_dataset,
                                                     validation_dataset=train_dataset)

        monkeypatch.setattr(pipeline, "accuracy_score", lambda y_true, y_pred: 1)
        sklearn_best_pipeline = pipeline.get_best_modeling_pipeline_type(transformer_modeling_pipeline=tf_pipeline,
                                                                         sklearn_modeling_pipeline=sklearn_pipeline,
                                                                         x_test=x_test,
                                                                         y_test=y_test,
                                                                         test_dataset=train_dataset)

        monkeypatch.setattr(tf_pipeline, "evaluate", lambda dataset, batch_size: 1)
        tf_best_pipeline = pipeline.get_best_modeling_pipeline_type(transformer_modeling_pipeline=tf_pipeline,
                                                                    sklearn_modeling_pipeline=sklearn_pipeline,
                                                                    x_test=x_test,
                                                                    y_test=y_test,
                                                                    test_dataset=train_dataset)

        assert isinstance(sklearn_best_pipeline, BaseEstimator)
        assert isinstance(tf_best_pipeline, TFDistilBertForSequenceClassification)

    def test_get_processing_pipeline(self, testing_config):
        pipeline = TextPipeline(testing_config)
        sklearn_pipeline = pipeline.get_processing_pipeline()
        tf_pipeline = pipeline.get_processing_pipeline(use_sklearn=False)

        assert isinstance(sklearn_pipeline, Pipeline)
        assert isinstance(tf_pipeline, TFDistilBertForSequenceClassification)

    def test_get_model(self, testing_config, structured_set, train_dataset):
        testing_config.PATIENCE = 3
        testing_config.DISEASES = ["test_disease1", "test_disease2", "test_disease3"]
        pipeline = TextPipeline(testing_config)

        x_train, x_test, y_train, y_test = structured_set

        pipeline.fit_structured_data(x_train, x_test, y_train, y_test)
        pipeline.fit_datasets(train_dataset, train_dataset, train_dataset)

        #sklearn_model = pipeline.get_model(x_train, x_test, y_train, y_test)
        tf_model = pipeline.get_model(use_sklearn=False)

    def test_get_modeling_pipeline(self, testing_config, structured_set, train_dataset):
        testing_config.PATIENCE = 3
        testing_config.DISEASES = ["test_disease1", "test_disease2", "test_disease3"]
        pipeline = TextPipeline(testing_config)

        x_train, x_test, y_train, y_test = structured_set

        pipeline.fit_structured_data(x_train, x_test, y_train, y_test)
        pipeline.fit_datasets(train_dataset, train_dataset, train_dataset)

        sklearn_pipeline = pipeline.get_modeling_pipeline(use_sklearn=True)
        tf_pipeline = pipeline.get_modeling_pipeline(use_sklearn=False,
                                                     train_dataset=train_dataset,
                                                     validation_dataset=train_dataset)
