from dermclass_structured import config
import logging
from typing import Union
import pandas as pd
import csv

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


def reorder_df(data: pd.DataFrame, list_to_reorder: list):
    """A function to order input DataFrame created from json file"""
    data = data.reindex(columns=list_to_reorder)
    return data


def reorder_df_from_file(data: pd.DataFrame, file_path: str = config.DATA_PATH):
    """A function to order input DataFrame created from json file using file"""
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if "target" in fieldnames:
            fieldnames.remove("target")
        reordered_data = reorder_df(data, fieldnames)

        return reordered_data
