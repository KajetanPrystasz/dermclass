import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

from dermclass_models.structured.processing.preprocessors import CastTypesTransformer
from dermclass_models.structured.config import StructuredConfig


class StructuredPpcPipeline:

    def __init__(self, config: StructuredConfig):
        self.config = config

        variables = config.NA_VALIDATION_VAR_DICT
        self.categorical_variables = variables["CATEGORICAL_NA_ALLOWED"] + variables["CATEGORICAL_NA_NOT_ALLOWED"]
        self.ordinal_variables = variables["ORDINAL_NA_ALLOWED"] + variables["ORDINAL_NA_NOT_ALLOWED"]
        self.numeric_variables = variables["NUMERIC_NA_ALLOWED"] + variables["NUMERIC_NA_NOT_ALLOWED"]
        self.all_variables = self.categorical_variables + self.ordinal_variables + self.numeric_variables


def fit_ppc_pipeline(self, x_train: pd.DataFrame) -> ColumnTransformer:
    """Fit provided x_train data to preprocessing data"""

    ppc_pipeline = ColumnTransformer(transformers=[
        ("Cast dtypes", CastTypesTransformer(categorical_variables=self.categorical_variables,
                                             ordinal_variables=self.ordinal_variables,
                                             numeric_variables=self.numeric_variables),
         self.all_variables),

        ("Fill_na_categorical", SimpleImputer(strategy='most_frequent'), self.variables["CATEGORICAL_NA_ALLOWED"]),
        ("Fill_na_ordinal", SimpleImputer(strategy='most_frequent'), self.variables["ORDINAL_NA_NOT_ALLOWED"]),
        ("Fill_na_numeric", SimpleImputer(strategy='median'), self.variables["NUMERIC_NA_ALLOWED"]),

        ("Encode ordinal", OrdinalEncoder(), self.ordinal_variables),
        ("Encode categorical", OneHotEncoder(), self.categorical_variables),

        ("Remove skewness", PowerTransformer(), self.numeric_variables),
        ("Scale data", RobustScaler(with_centering=False), self.ordinal_variables + self.numeric_variables)],
        remainder="passthrough")

    ppc_pipeline_fitted = ppc_pipeline.fit(x_train)
    self.logger.info("Successfully fitted the preprocessing pipeline")
    return ppc_pipeline_fitted
