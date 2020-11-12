# TODO: Add isinstance type blockers

import logging
import abc

import pandas as pd

from dermclass_models.base.config import BaseConfig


class ValidationError(BaseException):
    pass


class Validation:

    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def reorder_df(self, df: pd.DataFrame, proper_order: list = None) -> pd.DataFrame:
        """A function to order input DataFrame created from json file into proper order"""
        if proper_order is None:
            proper_order = self.config.VARIABLE_ORDER
        df = df.reindex(columns=proper_order)
        return df

    def validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """A function used to check if all columns match expected columns"""
        expected_columns = self.config.VARIABLE_ORDER
        for validated_column in df.columns:
            if validated_column not in expected_columns:
                raise ValidationError(f"Column {validated_column} not in expected_columns!")
        self.logger.info("Successfully validated input data")
        return df

    @abc.abstractmethod
    def custom_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
