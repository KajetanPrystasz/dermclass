import logging

import pandas as pd
import numpy as np

from dermclass_models.base.pickle import Pickle
from dermclass_models.base.processing.validation import Validation
from dermclass_models import __version__ as _version


class Predict:

    def __init__(self, config):
        self.config = config

        self.pipeline_type = self.config.pipeline_type
        self.pipeline_version = _version

        self.logger = logging.getLogger(__name__)

        self.validator = Validation(self.config)
        self.pickler = Pickle(self.config)

    def make_prediction(self, input_dict: dict) -> np.ndarray:
        """Make for the input_data"""

        if isinstance(input_dict, dict):
            df = pd.DataFrame([input_dict])
        else:
            raise TypeError("Wrong data input")

        df_valid = self.validator.validate_columns(df)
        df_reordered = self.validator.reorder_df(df_valid)
        df_final = self.validator.custom_validation(df_reordered)

        pipeline_file_name = f"{self.config.PIPELINE_NAME}_{_version}.pkl"
        pipeline = self.pickler.load_pipeline(pipeline_file_name)

        prediction = pipeline.predict(df_final)

        self.logger.info(f"Made predictions with model version: {_version}"
                         f"Inputs: {df_final} "
                         f"Predictions: {prediction}")

        return prediction
