from typing import Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import spacy

from sklearn.base import TransformerMixin, BaseEstimator

from dermclass_models.base.processing.preprocessors import Preprocessors
from dermclass_models.base.config import BaseConfig

from dermclass_models.text.config import TextConfig


class TextPreprocessors(Preprocessors):

    def __init__(self, config: BaseConfig = TextConfig):
        super().__init__(config)

    def load_class_from_dir(self, path: Path) -> pd.DataFrame:
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

        self.logger.info(f"Successfully loaded class {class_name}")
        return df

    def load_dataset_from_dir(self, path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        df = pd.DataFrame()

        for class_dir in path.iterdir():
            class_df = self.load_class_from_dir(class_dir)
            df = pd.concat([df, class_df])
        df = df.reset_index().drop("index", axis=1)

        self.logger.info("Successfully loaded the data")
        return df

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:

        path = self.config.DATA_PATH
        s_ppc = TextPreprocessors(self.config)
        df = s_ppc.load_dataset_from_dir(path)

        return df


# TODO: Think about making it faster (no for loop on rows)
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
