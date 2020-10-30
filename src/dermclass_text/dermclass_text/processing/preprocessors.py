import logging
import spacy
import numpy as np

from dermclass_structured import config
import pandas as pd
from typing import Tuple

from sklearn.base import TransformerMixin, BaseEstimator

_logger = logging.getLogger(__name__)


def split_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split x and y data from given Pandas DataFrame"""

    x = df.drop(config.variables["TARGET"], 1)
    y = df[config.variables["TARGET"]]
    _logger.info("Successfully splat the data")
    return x, y, df


def load_class_from_dir(path: str) -> pd.DataFrame:
    """Load data from provided path"""
    class_name = path.name

    df = pd.DataFrame()
    index_num = [0]
    for child in path.iterdir():
        with open(child, "r", encoding='utf-8') as file:
            text = file.read().replace("\n", " ")
            file_df = pd.DataFrame({"target": class_name,
                                    "text": text},
                                   index=index_num)
            index_num[0] = index_num[0] + 1
            df = pd.concat([df, file_df], axis=0)

    _logger.info(f"Successfully loaded class {class_name}")
    return df


def load_dataset_from_dir(path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.DataFrame()

    for class_dir in path.iterdir():
        class_df = load_class_from_dir(class_dir)
        df = pd.concat([df, class_df])
    df = df.reset_index().drop("index", axis=1)

    _logger.info("Successfully loaded the data")
    x, y, df_final = split_target(df)

    return x, y, df_final


class SpacyPreprocessor(TransformerMixin, BaseEstimator):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.doc = None
        self.x = None

    def fit(self, x, y=None):
        self.x = x
        return self

    def transform(self, x=None, y=None):
        if x is None:
            x = self.x

        array = np.array([])
        for row in x["text"]:
            tokens = self.nlp(row)
            clean_tokens = [token for token in tokens if not (token.is_punct or token.is_stop)]
            lemmas = [token.lemma_ for token in clean_tokens]
            text_lemmatized = " ".join(lemmas)
            array = np.append(array, text_lemmatized)

        return array
