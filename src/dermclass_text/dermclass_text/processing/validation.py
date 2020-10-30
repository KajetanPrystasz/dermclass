from dermclass_text import config
import logging
import pandas as pd
import csv

_logger = logging.getLogger(__name__)


def validate_inputs():
    pass


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

