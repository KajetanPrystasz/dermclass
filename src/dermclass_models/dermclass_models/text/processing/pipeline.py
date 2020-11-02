import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from dermclass_models.base.processing.pipeline import PpcPipeline

from dermclass_models.text.config import TextConfig
from dermclass_models.text.processing.preprocessors import SpacyPreprocessor


class TextPpcPipeline(PpcPipeline):

    def __init__(self, config: TextConfig = TextConfig):

        super().__init__(config)

        self.variables = [var for var in config.VARIABLE_ORDER if var != self.config.TARGET]

    def fit_ppc_pipeline(self, x_train: pd.DataFrame = None) -> Pipeline:
        """Fit provided x_train data to preprocessing data"""
        if x_train is None:
            x_train = self.x_train

        ppc_pipeline = Pipeline([("Lemmatization, punctuation and stopwords removal",
                                  SpacyPreprocessor()),
                                 ("Tfidf Vectorizer", TfidfVectorizer(ngram_range=(1, 2)))])

        ppc_pipeline_fitted = ppc_pipeline.fit(x_train)
        self.logger.info("Successfully fitted the preprocessing pipeline")
        return ppc_pipeline_fitted
