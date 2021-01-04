import abc
import logging
from pathlib import Path
from typing import Union, Tuple, List

import pandas as pd
import numpy as np

import tensorflow as tf

from dermclass_models.config import StructuredConfig, ImageConfig, TextConfig
from dermclass_models.persistence import BasePersistence
from dermclass_models.validation import TextValidation, StructuredValidation, validate_variables
from dermclass_models.pipeline import TransformersModelingPipeline

from dermclass_models import __version__ as dermclass_models_version

from sklearn.pipeline import Pipeline as SklearnPipeline

DataFrame = pd.DataFrame
Sequential = tf.keras.models.Sequential


class _BasePrediction(abc.ABC):

    def __init__(self, config):
        """
        Abstract base class used for making prediction
        :param config: Config object for the class
        """
        validate_variables(config)

        self.config = config
        self.logger = logging.getLogger(__name__)
        self.persister = BasePersistence(config)
        self.modeling_pipeline = None
        self.backend = None
        self.path = None

    def load_pipeline(self, backend: str = None, path: Path = None):
        """Function to load pipeline using persister and fit it as a modeling pipeline
        :param backend: Type of backend used for loading given pipeline, has to be one of ["joblib", "tf", "tfm"]
        :param path: Path to loaded file or directory
        :return: Returns a modeling pipeline to make predictions with
        """
        backend = backend or self.backend
        validate_variables(backend)
        if not self.persister:
            raise RuntimeError("No preprocessor object fitted")

        modeling_pipeline = self.persister.load_pipeline(backend=backend, path=path)
        self.modeling_pipeline = modeling_pipeline
        return modeling_pipeline


class _SklearnPrediction(_BasePrediction):

    def __init__(self, config):
        super().__init__(config)
        self.backend = "joblib"

    def _make_sklearn_prediction(self, data: dict) -> Tuple[np.ndarray, str]:
        """
        Utility function for making predictions with sklearn
        :param data: Data to make predictions on
        :return: Returns a tuple of class probabilities and str with class prediction
        """
        modeling_pipeline = self.modeling_pipeline or self.load_pipeline(self.path)
        validate_variables(modeling_pipeline, data)

        prediction = modeling_pipeline.predict_proba(data)[0]
        max_pred_idx = np.argmax(prediction)
        prediction_proba = prediction[max_pred_idx]

        try:
            map_ = self.config.LABEL_MAPPING
        except AttributeError:
            map_ = self.config.DISEASES
        prediction_string = map_[max_pred_idx]

        self.logger.info(f"Made predictions with model version: {dermclass_models_version} "
                         f"Inputs: {data} "
                         f"Prediction: {prediction_string} "
                         f"Probability: {prediction_proba}")
        return prediction_proba, prediction_string


class _TfPrediction(_BasePrediction):

    def _make_tf_prediction(self, data: np.array, diseases=List[str]) -> Tuple[np.array, str]:
        """
        Utility function for making predictions with tensorflow
        :param data: Data to make predictions in numpy ndarray format
        :param diseases: A list of diseases names in proper order
        :return: Returns a tuple of class probabilities and str with class prediction
        """
        modeling_pipeline = self.modeling_pipeline or self.load_pipeline(self.path)
        diseases = diseases or self.config.DISEASES
        validate_variables(modeling_pipeline, data, diseases)

        prediction = modeling_pipeline.predict(data)[0]
        max_pred_idx = np.argmax(prediction)
        prediction_proba = prediction[max_pred_idx]

        try:
            map_ = self.config.LABEL_MAPPING
        except AttributeError:
            map_ = self.config.DISEASES
        prediction_string = map_[max_pred_idx]

        self.logger.info(f"Made predictions with model version: {dermclass_models_version} "
                         f"Inputs: {data} "
                         f"Prediction: {prediction_string} "
                         f"Probability: {prediction_proba}")
        return prediction, prediction_string


class ImagePrediction(_TfPrediction):
    def __init__(self, config: ImageConfig = ImageConfig):
        super().__init__(config)
        self.backend = "tf"
        self.img_shape = None

    def _get_img_shape(self, modeling_pipeline: Sequential) -> Tuple[int, int]:
        """
        Utility function to get image shape, necessary for resizing input data
        :param modeling_pipeline: A tensorflow model object to get image shape from
        :return: A tuple with image shape
        """
        validate_variables(modeling_pipeline)
        if modeling_pipeline.layers[1].name == "efficientnetb7":
            img_size = (600, 600, 3)
        elif modeling_pipeline.layers[1].name == "efficientnetb6":
            img_size = (528, 528, 3)
        else:
            img_size = (456, 456, 3)

        self.img_size = img_size
        return img_size

    def _prepare_data(self, input_data: dict, img_shape: Tuple[int, int]) -> np.array:
        """
        Utility function to prepare data to format which can be used in modeling pipeline
        :param input_data: Input data to make prediction on
        :param img_shape: Shape of image to resize data
        :return: An array with data ready for making predictions using modeling pipeline
        """
        img_shape = img_shape or self.img_shape
        validate_variables(input_data, img_shape)

        data = input_data["image_array"]
        data = np.resize(data, img_shape)
        data = np.expand_dims(data, 0)

        return data

    def make_prediction(self, input_data: dict, img_shape: Tuple[int, int] = None) -> np.array:
        """
        Function to make prediction on given data
        :param input_data: Input data to make prediction on
        :param img_shape: Shape of image to resize data
        :return: Returns a tuple of class probabilities and str with class prediction
        """
        self.modeling_pipeline = self.modeling_pipeline or self.load_pipeline(self.path)
        img_shape = img_shape or self._get_img_shape(self.modeling_pipeline)
        validate_variables(input_data, img_shape)
        data = self._prepare_data(input_data, img_shape)

        prediction_probabilities, prediction_string = self._make_tf_prediction(data)
        return prediction_probabilities, prediction_string


class TextPrediction(_TfPrediction, _SklearnPrediction):
    def __init__(self, config: TextConfig = TextConfig):
        super().__init__(config)
        self.backend = self._get_backend()
        self.validator = TextValidation(config)

    def _get_backend(self, pickle_dir: Path = None, pipeline_type: str = None) -> str:
        """
        Utility function to get type of backend that should be used in the predictions
        :param pickle_dir: Directory which should be iterated to find backend
        :param pipeline_type: A string indicating type of pipeline to get a config object name from
        :return: A backend for making predictions
        """
        pickle_dir = pickle_dir or self.config.PICKLE_DIR
        pipeline_type = pipeline_type or self.config.PIPELINE_TYPE
        validate_variables(pickle_dir, pipeline_type)

        for file in pickle_dir.iterdir():
            try:
                if file.name.startswith(pipeline_type):
                    backend = file.name.split(".")[-1]
            except FileNotFoundError:
                None
        return backend

    def _prepare_data(self, input_data: dict) -> Union[DataFrame, np.array]:
        """
        Utility function to prepare data to format and validate data which can be used in modeling pipeline
        :param input_data: Input data to make prediction on
        :return: An array or pandas DataFrame with data ready for making predictions using modeling pipeline
        """
        validate_variables(input_data)
        if not self.validator:
            raise RuntimeError("No validator object fitted")

        df = pd.DataFrame(input_data, index=[0])
        df_validated = self.validator.validate(df)
        return df_validated

    def make_prediction(self, input_data: dict) -> Tuple[np.ndarray, str]:
        """
        Function to make prediction on given data
        :param input_data: Input data to make prediction on
        :return: Returns a tuple of class probabilities and str with class prediction
        """
        validate_variables(input_data)
        data = self._prepare_data(input_data)

        if self.backend == "joblib":
            prediction_probabilities, prediction_string = self._make_sklearn_prediction(data)
        else:
            prediction_probabilities, prediction_string = self._make_tf_prediction(data)
        return prediction_probabilities, prediction_string


class StructuredPrediction(_SklearnPrediction):
    def __init__(self, config: StructuredConfig = StructuredConfig):
        super().__init__(config)
        self.validator = StructuredValidation(config)

    def _prepare_data(self, input_data: dict) -> DataFrame:
        """
        Utility function to prepare data to format and validate data which can be used in modeling pipeline
        :param input_data: Input data to make prediction on
        :return: Returns a pandas DataFrame with data ready for making predictions using modeling pipeline
        """
        validate_variables(input_data)
        if not self.validator:
            raise RuntimeError("No validator object fitted")

        df = pd.DataFrame(input_data, index=[0])
        df_validated = self.validator.validate(df)
        return df_validated

    def make_prediction(self, input_data: dict) -> Tuple[np.ndarray, str]:
        """
        Function to make prediction on given data
        :param input_data: Input data to make prediction on
        :return: Returns a tuple of class probabilities and str with class prediction
        """
        validate_variables(input_data)

        data = self._prepare_data(input_data)
        prediction_probabilities, prediction_string = self._make_sklearn_prediction(data)
        return prediction_probabilities, prediction_string
