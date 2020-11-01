from typing import Tuple, List

import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator

from dermclass_models.base.processing.preprocessors import Preprocessors

from dermclass_models.structured.config import StructuredConfig


class StructuredPreprocessors(Preprocessors):

    def __init__(self, config: StructuredConfig):
        super().__init__(config)

    def load_csv(self, path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Load data from provided path"""

        df = pd.read_csv(path)
        x, y = self.split_target(df)
        self.logger.info("Successfully loaded the data")
        return x, y, df

    # TODO: Fix this
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:

        path = self.config.DATA_PATH
        s_ppc = StructuredPreprocessors(self.config)
        x, y, df = s_ppc.load_csv(path)
        return x, y, df


class CastTypesTransformer(TransformerMixin, BaseEstimator):
    """"This transformer cast types of Pandas DataFrame to according types"""

    def __init__(self, categorical_variables: List[str], ordinal_variables: List[str],
                 numeric_variables: List[str]):
        self.categorical_variables = categorical_variables
        self.ordinal_variables = ordinal_variables
        self.numeric_variables = numeric_variables

        self.x_cat = None
        self.x_ord = None
        self.x_num = None

        self.x = None
        self.y = None

    def fit(self, x, y=None):
        self.x = x
        self.y = y
        return self

    def transform(self, x=None, y=None):
        if x is None:
            x = self.x
        x_cat = x[self.categorical_variables]
        x_ord = x[self.ordinal_variables]
        x_num = x[self.numeric_variables]

        self.x_cat = x_cat.astype("category")
        self.x_ord = x_ord.astype("int")
        self.x_num = x_num.astype("float32")

        self.x = pd.concat([self.x_cat, self.x_ord, self.x_num], axis=1)
        return self.x
