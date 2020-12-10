import abc
import logging
from pathlib import Path
from typing import Union, Tuple

import pandas as pd
import numpy as np

import tensorflow as tf

from dermclass_models2.config import StructuredConfig, ImageConfig, TextConfig
from dermclass_models2.persistence import BasePersistence
from dermclass_models2 import __version__ as dermclass_models_version

DataFrame = pd.DataFrame
Sequential = tf.keras.models.Sequential


class _BasePrediction(abc.ABC):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.persister = BasePersistence(config)
        self.modeling_pipeline = None
        self.backend = None
        self.path = None

    def load_pipeline(self, backend: str = None, path: Path = None) -> Union[Sequential]:
        path = path or self.path
        backend = backend or self.backend
        modeling_pipeline = self.persister.load_pipeline(backend=backend, path=path)
        self.modeling_pipeline = modeling_pipeline
        return modeling_pipeline


class _SklearnPrediction(_BasePrediction):

    def _make_sklearn_prediction(self, data: dict) -> Tuple[np.ndarray, str]:
        modeling_pipeline = self.modeling_pipeline or self.load_pipeline(self.path)
        prediction = modeling_pipeline.predict(data)
        prediction_probabilities = prediction[0]
        prediction_string = prediction[1]

        self.logger.info(f"Made predictions with model version: {dermclass_models_version}"
                         f"Inputs: {data} "
                         f"Prediction: {prediction_string}"
                         f"Probability: {prediction_probabilities}")
        return prediction_probabilities, prediction_string


class _TfPrediction(_BasePrediction):

    def _make_tf_prediction(self, data: np.ndarray) -> Tuple[np.ndarray, str]:
        modeling_pipeline = self.modeling_pipeline or self.load_pipeline(self.path)

        prediction_probabilities = modeling_pipeline.predict(data)
        prediction_string = self.config.DISEASES[prediction_probabilities.argmax()]
        self.logger.info(f"Made predictions with model version: {dermclass_models_version}"
                         f"Inputs: {data} "
                         f"Prediction: {prediction_string}"
                         f"Probability: {prediction_probabilities}")
        return prediction_probabilities, prediction_string


class ImagePrediction(_TfPrediction):
    def __init__(self, config: ImageConfig = ImageConfig):
        super().__init__(config)
        self.backend = "tf"
        self.img_shape = None

    def _get_img_shape(self, modeling_pipeline) -> Tuple[int, int]:
        if isinstance(modeling_pipeline, tf.keras.applications.EfficientNetB7):
            img_size = (600, 600)
        elif isinstance(modeling_pipeline, tf.keras.applications.EfficientNetB6):
            img_size = (528, 528)
        else:
            img_size = (456, 456)

        self.img_size = img_size
        return img_size

    def _prepare_data(self, input_data: dict, img_shape: Tuple[int, int]):
        img_shape = img_shape or self.img_shape
        data = input_data["image"]
        data = np.resize(data, img_shape)
        data = np.expand_dims(data, 0)
        return data

    def make_prediction(self, input_data: dict, img_shape: Tuple[int, int]) -> np.array:
        img_shape = img_shape or self._get_img_shape(self.modeling_pipeline)
        data = self._prepare_data(input_data, img_shape)
        prediction_probabilities, prediction_string = self._make_tf_prediction(data)
        return prediction_probabilities, prediction_string


class TextPrediction(_TfPrediction, _SklearnPrediction):
    def __init__(self, config: TextConfig = TextConfig):
        super().__init__(config)
        self.backend = self._get_backend()

    def _get_backend(self, pickle_dir: Path = None) -> str:
        pickle_dir = pickle_dir or self.config.PICKLE_DIR

        for file in pickle_dir.iterdir():
            if file.name.startswith(self.config.PIPELINE_TYPE):
                backend = file.name.split(".")[-1]
        return backend

    def _prepare_data(self, input_data: dict):
        data = input_data["text"]
        if self.backend == "joblib":
            data = pd.DataFrame(input_data, index=[0]).drop("label", axis=1)
        else:
            data = data
        return data

    def make_prediction(self, input_data: dict) -> Tuple[np.ndarray, str]:
        data = self._prepare_data(input_data)
        if self.backend == "joblib":
            prediction_probabilities, prediction_string = self._make_sklearn_prediction(data)
        else:
            prediction_probabilities, prediction_string = self._make_tf_prediction(data)
        return prediction_probabilities, prediction_string


class StructuredPrediction(_SklearnPrediction):
    def __init__(self, config: StructuredConfig = StructuredConfig):
        super().__init__(config)

    @staticmethod
    def _prepare_data(input_data: dict) -> DataFrame:
        data = pd.DataFrame(input_data, index=[0]).drop("label", axis=1)
        return data

    def make_prediction(self, input_data: dict) -> Tuple[np.ndarray, str]:
        data = self._prepare_data(input_data)
        prediction_probabilities, prediction_string = self._make_sklearn_prediction(data)
        return prediction_probabilities, prediction_string
