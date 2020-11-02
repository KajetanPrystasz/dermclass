from typing import Tuple
import logging
import pandas as pd
import abc

from dermclass_models.base.config import BaseConfig


class Preprocessors:

    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def split_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split x and y data from given Pandas DataFrame"""
        x = df.drop(self.config.TARGET, 1)
        y = df[self.config.TARGET]
        self.logger.info("Successfully splat the data")
        return x, y

    # TODO: Add way of handling func with path instead of self
    @abc.abstractmethod
    def load_data(self):
        df = pd.DataFrame
        return df
