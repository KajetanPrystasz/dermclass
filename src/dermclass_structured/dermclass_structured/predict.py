from dermclass_structured.pickles.pickle_handling import load_pipeline
from dermclass_structured.processing.validation import validate_inputs, reorder_df_from_file
from dermclass_structured import config, __version__ as _version
import pandas as pd
import logging
from typing import Union
import numpy as np

_logger = logging.getLogger(__name__)


def make_prediction(input_data: Union[dict, pd.DataFrame]) -> np.ndarray:
    """Make for the input_data"""

    if isinstance(input_data, dict):
        data = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        data = input_data
    else:
        raise TypeError("Wrong data input")

    data = reorder_df_from_file(validate_inputs(data))

    pipeline_file_name = f"{config.PIPELINE_NAME}_{_version}.pkl"
    pipeline = load_pipeline(file_name=pipeline_file_name)
    prediction = pipeline.predict(data)

    _logger.info(f"Made predictions with model version: {_version} "
                 f"Inputs: {data} "
                 f"Predictions: {prediction}")
    return prediction
