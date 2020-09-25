from dermclass_structured import config
import logging
from typing import Union
import pandas as pd

_logger = logging.getLogger(__name__)


def validate_inputs(input_data: Union[dict, pd.DataFrame]) -> Union[dict, pd.DataFrame]:
    """Validate input data by dropping rows with unexpected NA'S"""

    validated_data = input_data.copy()

    if input_data[config.variables["NUMERIC_NA_NOT_ALLOWED"]].isnull().any().any():
        validated_data = validated_data.dropna(axis=0, subset=config.variables["NUMERIC_NA_NOT_ALLOWED"])

    if input_data[config.variables["CATEGORICAL_NA_NOT_ALLOWED"]].isnull().any().any():
        validated_data = validated_data.dropna(axis=0, subset=config.variables["CATEGORICAL_NA_NOT_ALLOWED"])

    _logger.info("Successfully validated input data")
    return validated_data
