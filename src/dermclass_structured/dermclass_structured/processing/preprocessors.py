import logging
from sklearn.base import TransformerMixin, BaseEstimator
from dermclass_structured import config
import pandas as pd
from typing import Tuple, List

_logger = logging.getLogger(__name__)


def split_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split x and y data from given Pandas DataFrame"""

    x = df.drop(config.variables["TARGET"], 1)
    y = df[config.variables["TARGET"]]
    _logger.info("Successfully splat the data")
    return x, y


def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load data from provided path"""

    df = pd.read_csv(path)
    x, y = split_target(df)
    _logger.info("Successfully loaded the data")
    return x, y, df


class CastTypesTransformer(TransformerMixin, BaseEstimator):
    """"This transformer cast types of Pandas DataFrame to according types"""

    def __init__(self, categorical_variables: List[str], ordinal_variables: List[str], numeric_variables: List[str]):
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
        _logger.info("Successfully transformed variables")
        return self.x
