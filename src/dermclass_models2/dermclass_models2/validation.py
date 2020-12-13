import abc
import logging
from typing import List

import pandas as pd

from dermclass_models2.config import StructuredConfig, TextConfig
DataFrame = pd.DataFrame


def validate_variables(**kwargs):
    for key, value in kwargs.items():
        try:
            if not isinstance(key, value):
                raise TypeError(f"Function cannot be run using {value} type for {key}")
        # Handle pandas DataFrames
        except ValueError:
            raise TypeError(f"Function cannot be run using {value} type for {key}")


class ValidationError(BaseException):
    pass


class _SklearnValidation(abc.ABC):

    def __init__(self, config):
        """
        Abstract class for validation with sklearn
        :param config: Config object for the class
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _reorder_df(self, df: DataFrame, proper_order: List[str] = None) -> DataFrame:
        """
        Utility function to reorder inputed data frame in proper order, necessary when loading json unordered dictionary
        since sklearn pipelines raises warnings if order of column changes
        :param df: A pandas DataFrame to reorder
        :param proper_order: A list of string columns with proper order
        :return: Returns a pandas DataFrame with reordered columns
        """
        if proper_order is None:
            proper_order = self.config.VARIABLE_ORDER
        df = df.reindex(columns=proper_order)
        return df

    def _validate_columns(self, df: DataFrame) -> DataFrame:
        """
        Utility function to check if one of the input columns aren't unexpected, raises error otherwise
        :param df: A pandas DataFrame to check columns
        :return: A pandas DataFrame with all columns expected
        """
        expected_columns = self.config.VARIABLE_ORDER
        for validated_column in df.columns:
            if validated_column not in expected_columns:
                raise ValidationError(f"Column {validated_column} not in expected_columns!")
        self.logger.info("Successfully validated input data")
        return df

    @abc.abstractmethod
    def validate(self, df: DataFrame) -> pd.DataFrame:
        """
        Abstract method for validation
        :param df: Input pandas DataFrame
        :return: Output validated pandas DataFrame
        """
        return df


class StructuredValidation(_SklearnValidation):

    def __init__(self, config: StructuredConfig = StructuredConfig):
        """
        Class for validating structured data
        :param config: Config object for the class
        """
        super().__init__(config)

    def _drop_unexpected_na(self, input_data: DataFrame) -> DataFrame:
        """
        Utility function to drop rows with unexpected NA values
        :param input_data: Input data to drop unexpected NA data
        :return: A pandas DataFrame with unexpected NA removed
        """
        validated_data = input_data.copy()

        if input_data[self.config.NA_VALIDATION_VAR_DICT["NUMERIC_NA_NOT_ALLOWED"]].isnull().any().any():
            validated_data = (validated_data
                              .dropna(axis=0,
                                      subset=self.config.NA_VALIDATION_VAR_DICT["NUMERIC_NA_NOT_ALLOWED"]))

        if input_data[self.config.NA_VALIDATION_VAR_DICT["CATEGORICAL_NA_NOT_ALLOWED"]].isnull().any().any():
            validated_data = (validated_data
                              .dropna(axis=0,
                                      subset=self.config.NA_VALIDATION_VAR_DICT["CATEGORICAL_NA_NOT_ALLOWED"]))

        self.logger.info("Successfully validated input data")
        return validated_data

    def validate(self, df: DataFrame) -> DataFrame:
        """
        Function to validate inputed pandas DataFrame
        :param df: A Pandas DataFrame to validate
        :return: Returns a validated pandas DataFrame
        """
        df = self._reorder_df(df)
        df = self._validate_columns(df)
        df = self._drop_unexpected_na(df)
        return df


class TextValidation(_SklearnValidation):

    def __init__(self, config: TextConfig = TextConfig):
        """
        Class for validating text data
        :param config: Config object for the class
        """
        super().__init__(config)

    def validate(self, df: DataFrame) -> DataFrame:
        """
        Function to validate inputed pandas DataFrame
        :param df: A Pandas DataFrame to validate
        :return: Returns a validated pandas DataFrame
        """
        df = self._reorder_df(df)
        df = self._validate_columns(df)
        return df
