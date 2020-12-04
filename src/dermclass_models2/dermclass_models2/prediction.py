import logging
from pathlib import Path

import abc

import pandas as pd
import numpy as np

from dermclass_models2.base.pickle import Pickle
from dermclass_models2.base.processing.validation import Validation
from dermclass_models2 import __version__ as _version


class Validator:
    def validate_input(*args, **kwargs):
        return args, kwargs

    # df_valid = self.validator.validate_columns(df)
    # df_reordered = self.validator.reorder_df(df_valid)
    # df_final = self.validator.custom_validation(df_reordered)


class _Prediction(abc.ABC):

    def __init__(self, config):
        self.config = config

        self.pipeline_version = _version

        self.logger = logging.getLogger(__name__)

        self.validator = Validation(self.config)
        self.persistor = Persistence(self.config)

    @abc.abstractmethod
    def _prepare_data_to_right_format(self):
        return pd.DataFrame

    @abc.abstractmethod
    def _make_prediction(self, pipeline, data):
        return int

    def _load_pipeline(self):
        pipeline_file_name = Path(f"{self.config.PIPELINE_TYPE}_{_version}")
        pipeline_path = self.config.PICKLE_DIR / pipeline_file_name
        pipeline = self.pickler.load_pipeline(pipeline_path)
        self.pipeline = pipeline
        return pipeline

    def predict(self, input_data: dict) -> np.ndarray:
        data = self._prepare_data_to_right_format(input_data)
        pipeline = self._load_pipeline
        prediction = self._make_prediction(pipeline, data)

        self.logger.info(f"Made predictions with model version: {_version}"
                         f"Inputs: {data} "
                         f"Predictions: {prediction}")
        return prediction


class _SklearnPrediction(abc.ABC, _Prediction):
    def _prepare_data_to_right_format_sklearn(self):
        pass

    def _make_prediction_sklearn(self):
        pass


class _TFPrediction(abc.ABC, _Prediction):
    def _prepare_data_to_right_format_tf(self):
        pass

    def _make_prediction_tf(self):
        pass


class StructuredPrediction(_SklearnPrediction):
    def _prepare_data_to_right_format(self):
        self._prepare_data_to_right_format_sklearn()

    def _make_prediction(self):
        self._prepare_data_to_right_format_sklearn()


class TextPrediction(_SklearnPrediction, _TFPrediction):
    def _prepare_data_to_right_format(self, formatt="sklearn"):
        if formatt == "sklearn":
            self._prepare_data_to_right_format_sklearn()
        else:
            self._prepare_data_to_right_format_tf()

    def _make_prediction(self, formatt="sklearn"):
        if formatt == "sklearn":
            self._make_prediction_sklearn()
        else:
            self._make_prediction_tf()


class ImagePrediction(_TFPrediction):
    def _prepare_data_to_right_format(self):
        self._prepare_data_to_right_format_tf()

    def _make_prediction(self):
        self._prepare_data_to_right_format_tf()
