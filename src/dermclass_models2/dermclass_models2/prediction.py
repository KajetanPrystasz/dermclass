import logging
from pathlib import Path

import abc

import pandas as pd
import numpy as np

from dermclass_models2.base.pickle import Pickle
from dermclass_models2.base.processing.validation import Validation
from dermclass_models2 import __version__ as _version

# TODO: Change gitingore for pickles and datasets


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
