import abc
import logging

import pandas as pd

from dermclass_models2.config import BaseConfig, StructuredConfig, TextConfig


# TODO: Add isinstance type blockers
class ValidationError(BaseException):
    pass


class _SklearnValidation(abc.ABC):

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
    def apply_custom_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class StructuredValidation(_SklearnValidation):

    def __init__(self, config: StructuredConfig = StructuredConfig):
        super().__init__(config)

    def drop_unexpected_na(self, input_data: pd.DataFrame) -> pd.DataFrame:

        validated_data = input_data.copy()

        if input_data[self.config.variables["NUMERIC_NA_NOT_ALLOWED"]].isnull().any().any():
            validated_data = validated_data.dropna(axis=0, subset=self.config.variables["NUMERIC_NA_NOT_ALLOWED"])

        if input_data[self.config.variables["CATEGORICAL_NA_NOT_ALLOWED"]].isnull().any().any():
            validated_data = validated_data.dropna(axis=0, subset=self.config.variables["CATEGORICAL_NA_NOT_ALLOWED"])

        self.logger.info("Successfully validated input data")
        return validated_data

    def apply_custom_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.drop_unexpected_na(df)
        return df


class TextValidation(_SklearnValidation):

    def __init__(self, config: TextConfig = TextConfig):
        super().__init__(config)

    def apply_custom_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.reorder_df(df)
        df = self.validate_columns(df)
        return df
