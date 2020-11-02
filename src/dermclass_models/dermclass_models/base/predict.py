import logging
from pathlib import Path

import pandas as pd
import numpy as np

from dermclass_models.base.pickle import Pickle
from dermclass_models.base.processing.validation import Validation
from dermclass_models import __version__ as _version


class Predict:

    def __init__(self, config):
        self.config = config

        self.pipeline_version = _version

        self.logger = logging.getLogger(__name__)

        self.validator = Validation(self.config)
        self.pickler = Pickle(self.config)

    def make_prediction(self, input_data: dict) -> np.ndarray:
        """Make for the input_data"""

        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            raise TypeError("Wrong data input")

        df_valid = self.validator.validate_columns(df)
        df_reordered = self.validator.reorder_df(df_valid)
        df_final = self.validator.custom_validation(df_reordered)

        pipeline_file_name = Path(f"{self.config.PIPELINE_TYPE}_{_version}.pkl")
        pipeline_path = self.config.PICKLE_DIR / pipeline_file_name
        pipeline = self.pickler.load_pipeline(pipeline_path)

        prediction = pipeline.predict(df_final)

        self.logger.info(f"Made predictions with model version: {_version}"
                         f"Inputs: {df_final} "
                         f"Predictions: {prediction}")

        return prediction
