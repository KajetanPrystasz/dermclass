import pandas as pd

from dermclass_models.base.processing.validation import Validation

from dermclass_models.structured.config import StructuredConfig


class StructuredValidation(Validation):

    def __init__(self, config: StructuredConfig = StructuredConfig):
        super().__init__(config)

    def custom_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.drop_unexpected_na(df)
        return df

    def drop_unexpected_na(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Validate input data by dropping rows with unexpected NA'S"""

        validated_data = input_data.copy()

        if input_data[self.config.variables["NUMERIC_NA_NOT_ALLOWED"]].isnull().any().any():
            validated_data = validated_data.dropna(axis=0, subset=self.config.variables["NUMERIC_NA_NOT_ALLOWED"])

        if input_data[self.config.variables["CATEGORICAL_NA_NOT_ALLOWED"]].isnull().any().any():
            validated_data = validated_data.dropna(axis=0, subset=self.config.variables["CATEGORICAL_NA_NOT_ALLOWED"])

        self.logger.info("Successfully validated input data")
        return validated_data
