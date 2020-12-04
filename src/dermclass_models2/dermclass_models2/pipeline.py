import abc
import logging
from typing import Type

import pandas as pd
import numpy as np

from optuna import create_study, Trial
from optuna.samplers import TPESampler

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse.csr import csr_matrix

from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast

import tensorflow as tf

from dermclass_models2.config import StructuredConfig, ImageConfig, TextConfig
from dermclass_models2.preprocessing import CastTypesTransformer, SpacyPreprocessor

from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB


# TODO: Poprawić obiektowość Liskov Substition  dla inputów do abstract class
class _SklearnModels(abc.ABC):

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.model = None
        self.processing_pipeline = None
        self.modeling_pipeline = None

        self.x_train = pd.DataFrame
        self.x_test = pd.DataFrame
        self.y_train = pd.Series
        self.y_test = pd.Series

    @abc.abstractmethod
    def get_processing_pipeline(self):
        pass

    @abc.abstractmethod
    def get_modeling_pipeline(self):
        pass

    def fit_data(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def _get_sklearn_model(self,
                           x_train: pd.DataFrame = None, x_test: pd.DataFrame = None,
                           y_train: pd.DataFrame = None, y_test: pd.DataFrame = None):
        x_train = x_train or self.x_train
        x_test = x_test or self.x_test
        y_train = y_train or self.y_train
        y_test = y_test or self.y_test

        processing_pipeline = self.get_processing_pipeline()
        processing_pipeline.fit(x_train)
        x_train = processing_pipeline.transform(x_train)
        x_test = processing_pipeline.transform(x_test)
        print(type(x_train))
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        model = self._tune_hyperparameters(self.config.trials_dict,
                                           x_train, x_test, y_train, y_test,
                                           **self.config.tuning_func_params)
        return model

    # TODO: Add additional metrics
    @staticmethod
    def _hyper_param_optimization(trial, model_name: str, trial_func: Trial,
                                  max_overfit: float, cv: int,
                                  x_train: pd.DataFrame, x_test: pd.DataFrame,
                                  y_train: pd.DataFrame, y_test: pd.DataFrame):
        model_obj = eval(model_name)
        cv_score = np.mean(cross_val_score(model_obj(**trial_func(trial)),
                                           x_train,
                                           y_train,
                                           scoring="accuracy",
                                           cv=cv))

        model = model_obj(**trial_func(trial))
        model.fit(x_train, y_train)

        train_score = accuracy_score(y_train, model.predict(x_train))
        test_score = accuracy_score(y_test, model.predict(x_test))
        if abs(train_score - test_score) > max_overfit:
            output = 0
        else:
            output = cv_score
        return output

    def _set_dfs(self, x_train, x_test, y_train, y_test):
        if isinstance(x_train, (pd.DataFrame, csr_matrix)):
            x_train = x_train
        else:
            x_train = self.x_train

        if isinstance(x_test, (pd.DataFrame, csr_matrix)):
            x_test = x_test
        else:
            x_test = self.x_test

        if isinstance(y_train, (pd.Series, csr_matrix)):
            y_train = y_train
        else:
            y_train = self.y_train

        if isinstance(y_test, (pd.Series, csr_matrix)):
            y_test = y_test
        else:
            y_test = self.y_test
        return x_train, x_test, y_train, y_test

    # TODO: Add pruning
    def _tune_hyperparameters(self, trials_dict,
                              x_train: pd.DataFrame = None, x_test: pd.DataFrame = None,
                              y_train: pd.DataFrame = None, y_test: pd.DataFrame = None,
                              max_overfit: float = 0.05, cv: int = 3, n_trials: int = 10, n_jobs: int = -1):

        x_train, x_test, y_train, y_test = self._set_dfs(x_train, x_test, y_train, y_test)
        self.studies = {}
        for model_name, trial_func in trials_dict.items():
            self.logger.info(f"Finding hyperparameters for {model_name}")
            sampler = TPESampler(seed=self.config.SEED)
            study = create_study(direction="maximize", study_name=model_name, sampler=sampler)
            study.optimize(lambda trial:
                           (self._hyper_param_optimization(trial, model_name, trial_func, max_overfit, cv,
                                                           x_train, x_test, y_train, y_test)),
                           n_trials=n_trials, n_jobs=n_jobs)

            self.studies[model_name] = study

        best_score, best_model, best_params = 0, self.config.DEFAULT_BEST_MODEL, {}
        for study in self.studies.values():
            if study.best_value > best_score:
                best_model = study.study_name
                best_score = study.best_value
                best_params = study.best_params
        self.logger.info(f"Best params found for: {best_model} with score: {best_score}")

        self.final_model = eval(best_model)(**best_params).fit(x_train, y_train)
        self.logger.info("Successfully tuned hyperparameters")
        return self.final_model


class _TfModels(abc.ABC):

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.model = None
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        self.model = None
        self.processing_pipeline = None
        self.modeling_pipeline = None

        self.learning_rate = 0.001
        self.metrics = ["accuracy"]

    @abc.abstractmethod
    def get_processing_pipeline(self):
        pass

    @abc.abstractmethod
    def get_model(self):
        return None

    @abc.abstractmethod
    def get_modeling_pipeline(self):
        pass

    def compile_model(self, model, learning_rate=None, metrics=None):
        learning_rate = learning_rate or self.learning_rate
        metrics = metrics or self.metrics
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=metrics)
        return model


class StructuredModels(_SklearnModels):

    def __init__(self, config: StructuredConfig = StructuredConfig):
        super().__init__(config)

        self.variables = config.NA_VALIDATION_VAR_DICT
        self.categorical_variables = (self.variables["CATEGORICAL_NA_ALLOWED"]
                                      + self.variables["CATEGORICAL_NA_NOT_ALLOWED"])
        self.ordinal_variables = self.variables["ORDINAL_NA_ALLOWED"] + self.variables["ORDINAL_NA_NOT_ALLOWED"]
        self.numeric_variables = self.variables["NUMERIC_NA_ALLOWED"] + self.variables["NUMERIC_NA_NOT_ALLOWED"]
        self.all_variables = self.categorical_variables + self.ordinal_variables + self.numeric_variables

    def get_processing_pipeline(self) -> Type[ColumnTransformer]:
        processing_pipeline = ColumnTransformer(transformers=[
            ("Cast dtypes", CastTypesTransformer(categorical_variables=self.categorical_variables,
                                                 ordinal_variables=self.ordinal_variables,
                                                 numeric_variables=self.numeric_variables),
             self.all_variables),

            ("Fill_na_categorical", SimpleImputer(strategy='most_frequent'),
             self.variables["CATEGORICAL_NA_ALLOWED"]),
            ("Fill_na_ordinal", SimpleImputer(strategy='most_frequent'), self.variables["ORDINAL_NA_NOT_ALLOWED"]),
            ("Fill_na_numeric", SimpleImputer(strategy='median'), self.variables["NUMERIC_NA_ALLOWED"]),

            ("Encode ordinal", OrdinalEncoder(), self.ordinal_variables),
            ("Encode categorical", OneHotEncoder(), self.categorical_variables),

            ("Remove skewness", PowerTransformer(), self.numeric_variables),
            ("Scale data", RobustScaler(with_centering=False), self.ordinal_variables + self.numeric_variables)],
            remainder="passthrough")

        self.processing_pipeline = processing_pipeline
        return processing_pipeline

    # TODO: Add note about having to transform_data first
    def get_model(self,
                  x_train: pd.DataFrame = None, x_test: pd.DataFrame = None,
                  y_train: pd.DataFrame = None, y_test: pd.DataFrame = None):

        model = self._get_sklearn_model(x_train, x_test, y_train, y_test)
        self.model = model
        return model

    def get_modeling_pipeline(self,
                              x_train: pd.DataFrame = None, x_test: pd.DataFrame = None,
                              y_train: pd.DataFrame = None, y_test: pd.DataFrame = None):
        processing_pipeline = self.get_processing_pipeline()
        model = self.get_model(x_train, x_test, y_train, y_test)

        modeling_pipeline = Pipeline([("Processing pipeline", processing_pipeline),
                                      ("Model", model)])
        self.modeling_pipeline = modeling_pipeline
        return modeling_pipeline


class _TransformersModelingPipeline:
    def __init__(self, processing_pipeline, model):
        self.processing_pipeline = processing_pipeline
        self.model = model

    def __call__(self, dataset, *args, **kwargs):
        dataset_encoded = self.processing_pipeline(dataset)
        predictions = self.model.predict(dataset_encoded)
        return predictions


# TODO: Refactor class to make it more logic friendly
class TextModels(_SklearnModels, _TfModels):

    def __init__(self, config: TextConfig = TextConfig):
        _SklearnModels.__init__(self, config)
        _TfModels.__init__(self, config)
        self.tokenizer = None

    def _encode_dataset(self, dataset):
        text_to_encode = []
        labels = []
        for batch in dataset.as_numpy_iterator():
            text_batch = batch[0]
            labels_batch = batch[1]

            text_batch_decoded = [text.decode("utf-8") for text in text_batch]
            text_to_encode += text_batch_decoded
            labels += labels_batch.tolist()
        text_encodings = self.tokenizer(text_to_encode, truncation=True, padding=True)
        text_batch_dataset = tf.data.Dataset.from_tensor_slices((
            dict(text_encodings),
            labels))
        return text_batch_dataset

    def get_model(self, use_sklearn=True, x_train=None, x_test=None, y_train=None, y_test=None):
        if use_sklearn:
            model = self._get_sklearn_model(x_train, x_test, y_train, y_test)
        else:
            model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        self.model = model
        return model

    def get_processing_pipeline(self, use_sklearn=True):
        if use_sklearn:
            processing_pipeline = Pipeline([("Lemmatization, punctuation and stopwords removal",
                                             SpacyPreprocessor()),
                                            ("Tfidf Vectorizer", TfidfVectorizer(ngram_range=(1, 2)))])
        else:
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            self.tokenizer = tokenizer
            processing_pipeline = self._encode_dataset

        self.processing_pipeline = processing_pipeline
        return processing_pipeline

    def get_modeling_pipeline(self, use_sklearn=True):
        processing_pipeline = self.get_processing_pipeline(use_sklearn)
        model = self.get_model(use_sklearn)
        if use_sklearn:
            modeling_pipeline = Pipeline([("Processing pipeline", processing_pipeline),
                                          ("Model", model)])
        else:
            model = self.compile_model(model)
            modeling_pipeline = _TransformersModelingPipeline(model=model,
                                                              processing_pipeline=processing_pipeline)

        self.modeling_pipeline = modeling_pipeline
        return modeling_pipeline


class ImageModels(_TfModels):

    def __init__(self, config: ImageConfig = ImageConfig):
        super().__init__(config)

        self.img_size = None
        self.model_obj = None

    def set_img_size_and_model_obj(self, img_size: tuple, model_obj: tf.keras.models.Sequential):
        self.img_size = img_size
        self.model_obj = model_obj

    def get_processing_pipeline(self, rescale=False):
        layers = [
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1)]

        if rescale:
            layers.append(tf.keras.layers.experimental.preprocessing.Rescaling(1 / 225))

        processing_pipeline = tf.keras.Sequential(layers)
        self.processing_pipeline = processing_pipeline
        return processing_pipeline

    # TODO: Add safeholder
    def get_model(self, model_obj=None):
        model_obj = model_obj or self.model_obj
        model = model_obj(include_top=False, weights='imagenet', classes=3)
        model.trainable = False

        self.model = model
        return model

    def get_modeling_pipeline(self, img_size=None, learning_rate=None, metrics=None):
        img_size = img_size or self.img_size

        processing_pipeline = self.get_processing_pipeline()
        model = self.get_model()

        modeling_pipeline = tf.keras.Sequential([tf.keras.Input(shape=img_size + (3,)),
                                                 processing_pipeline,
                                                 model,
                                                 tf.keras.layers.GlobalAveragePooling2D(),
                                                 tf.keras.layers.Dense(3, "softmax")
                                                 ])

        # TODO: Add weighted Adam
        self.compile_model(modeling_pipeline)

        self.modeling_pipeline = modeling_pipeline
        return modeling_pipeline
