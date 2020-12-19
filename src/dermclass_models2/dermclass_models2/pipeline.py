import abc
import logging
from typing import Union, Tuple, List
from pathlib import Path

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
from dermclass_models2.validation import validate_variables

from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB

DataFrame = pd.DataFrame
Series = pd.Series
Dataset = tf.data.Dataset
Sequential = tf.keras.models.Sequential


# TODO: Add input safeholders for get model function to make them not run if self.x_train etc. are Nones
class _SklearnPipeline(abc.ABC):

    def __init__(self, config):
        validate_variables(config)

        self.config = config
        self.logger = logging.getLogger(__name__)

        self.model = None
        self.processing_pipeline = None
        self.modeling_pipeline = None

        self.x_train = pd.DataFrame
        self.x_test = pd.DataFrame
        self.y_train = pd.Series
        self.y_test = pd.Series

    def fit_structured_data(self, x_train: DataFrame, x_test: DataFrame, y_train: Series, y_test: Series):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def _get_sklearn_model(self,
                           x_train: DataFrame = None, x_test: DataFrame = None,
                           y_train: Series = None, y_test: Series = None):
        x_train, x_test, y_train, y_test = self._set_dfs(x_train, x_test, y_train, y_test)
        validate_variables(x_train, x_test, y_train, y_test)

        processing_pipeline = self.get_processing_pipeline()
        processing_pipeline.fit(x_train)
        x_train = processing_pipeline.transform(x_train)
        x_test = processing_pipeline.transform(x_test)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        model = self._tune_hyperparameters(self.config.TRIALS_DICT,
                                           x_train, x_test, y_train, y_test,
                                           **self.config.TUNING_FUNC_PARAMS)
        return model

    @staticmethod
    def _hyper_param_optimization(trial, model_name: str, trial_func: Trial,
                                  max_overfit: float, cv: int,
                                  x_train: DataFrame, x_test: DataFrame,
                                  y_train: Series, y_test: Series):
        validate_variables(trial, model_name, trial_func, x_train, x_test, y_train, y_test)

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

    # TODO: To remove
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

    # TODO: To remove
    def _set_dfs_test(self, x_test, y_test):
        if isinstance(x_test, (pd.DataFrame, csr_matrix)):
            x_test = x_test
        else:
            x_test = self.x_test

        if isinstance(y_test, (pd.Series, csr_matrix)):
            y_test = y_test
        else:
            y_test = self.y_test
        return x_test, y_test

    # TODO: Add pruning
    def _tune_hyperparameters(self, trials_dict,
                              x_train: DataFrame = None, x_test: DataFrame = None,
                              y_train: Series = None, y_test: Series = None,
                              max_overfit: float = 0.05, cv: int = 3, n_trials: int = 10, n_jobs: int = -1):
        validate_variables(trials_dict,
                           x_train, x_test, y_train, y_test,
                           cv, n_trials, n_jobs)

        x_train, x_test, y_train, y_test = self._set_dfs(x_train, x_test, y_train, y_test)
        self.studies = {}
        for model_name, trial_func in trials_dict.items():
            self.logger.info(f"Finding hyperparameters for {model_name}")
            sampler = TPESampler(seed=self.config.SEED)
            study = create_study(direction="maximize", study_name=model_name, sampler=sampler)
            study.optimize(lambda trial:
                           (self._hyper_param_optimization(trial, model_name, trial_func, max_overfit, cv,
                                                           x_train, x_test, y_train, y_test)),
                           n_trials=n_trials, n_jobs=n_jobs,
                           gc_after_trial=True)

            self.studies[model_name] = study

        best_score, best_model, best_params = 0, self.config.DEFAULT_BEST_MODEL, {}
        for study in self.studies.values():
            if study.best_value > best_score:
                best_model = study.study_name
                best_score = study.best_value
                best_params = study.best_params
        self.logger.info(f"Best params found for: {best_model} with score: {best_score}")

        model = eval(best_model)(**best_params).fit(x_train, y_train)
        self.model = model

        self.logger.info("Successfully tuned hyperparameters")
        return model


class _TfPipeline(abc.ABC):

    def __init__(self, config):
        validate_variables(config)

        self.config = config
        self.logger = logging.getLogger(__name__)

        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config.PATIENCE)

        self.train_dataset = Dataset
        self.validation_dataset = Dataset
        self.test_dataset = Dataset

        self.model = None
        self.processing_pipeline = None
        self.modeling_pipeline = None

        self.learning_rate = 0.001

    def fit_datasets(self, train_dataset: Dataset, validation_dataset: Dataset, test_dataset: Dataset):
        validate_variables(train_dataset, validation_dataset, test_dataset)

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        validate_variables(train_dataset, validation_dataset, test_dataset)

    def _compile_model(self, model, learning_rate=None, metrics=None):
        model = model or self.model
        learning_rate = learning_rate or self.learning_rate
        metrics = metrics or self.config.METRICS
        validate_variables(model, learning_rate, metrics)

        if isinstance(model, TFDistilBertForSequenceClassification):
            loss = model.compute_loss
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=loss,
                      metrics=metrics)
        self.model = model

        self.logger.info("Successfully compiled tf model")
        return model

    def _train_model(self,
                     model: Sequential = None,
                     train_dataset: Dataset = None,
                     validation_dataset: Dataset = None,
                     n_epochs: int = None):
        model = model or self.model
        train_dataset = train_dataset or self.train_dataset
        validation_dataset = validation_dataset or self.validation_dataset
        n_epochs = n_epochs or self.config.NUM_EPOCHS
        validate_variables(model, train_dataset, validation_dataset, n_epochs)

        if isinstance(model, TFDistilBertForSequenceClassification):
            model.fit(train_dataset.batch(self.config.BATCH_SIZE),
                      validation_data=validation_dataset.batch(self.config.BATCH_SIZE),
                      epochs=n_epochs)
        else:
            model.fit(train_dataset,
                      validation_data=validation_dataset,
                      epochs=n_epochs)

        self.model = model

        self.logger.info("Successfully trained tf model")
        return model


class StructuredPipeline(_SklearnPipeline):

    def __init__(self, config: StructuredConfig = StructuredConfig):
        super().__init__(config)

        self.variables = config.NA_VALIDATION_VAR_DICT
        self.categorical_variables = (self.variables["CATEGORICAL_NA_ALLOWED"]
                                      + self.variables["CATEGORICAL_NA_NOT_ALLOWED"])
        self.ordinal_variables = self.variables["ORDINAL_NA_ALLOWED"] + self.variables["ORDINAL_NA_NOT_ALLOWED"]
        self.numeric_variables = self.variables["NUMERIC_NA_ALLOWED"] + self.variables["NUMERIC_NA_NOT_ALLOWED"]
        self.all_variables = self.categorical_variables + self.ordinal_variables + self.numeric_variables

    def get_processing_pipeline(self) -> ColumnTransformer:
        processing_pipeline = ColumnTransformer(transformers=[
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

        self.processing_pipeline = processing_pipeline

        self.logger.info("Successfully loaded structured pipeline")
        return processing_pipeline

    def get_model(self,
                  x_train: DataFrame = None, x_test: DataFrame = None,
                  y_train: Series = None, y_test: Series = None):
        x_train, x_test, y_train, y_test = self._set_dfs(x_train, x_test, y_train, y_test)
        validate_variables(x_train, x_test, y_train, y_test)

        model = self._get_sklearn_model(x_train, x_test, y_train, y_test)
        self.model = model

        self.logger.info("Successfully loaded structured model")
        return model

    def get_modeling_pipeline(self,
                              x_train: DataFrame = None, x_test: DataFrame = None,
                              y_train: Series = None, y_test: Series = None):
        x_train, x_test, y_train, y_test = self._set_dfs(x_train, x_test, y_train, y_test)

        processing_pipeline = self.get_processing_pipeline()
        model = self.get_model(x_train, x_test, y_train, y_test)
        validate_variables(x_train, x_test, y_train, y_test, processing_pipeline, model)

        modeling_pipeline = Pipeline([("Processing pipeline", processing_pipeline),
                                      ("Model", model)])
        modeling_pipeline.fit(x_train, y_train)

        self.modeling_pipeline = modeling_pipeline
        self.logger.info("Successfully loaded structured modeling pipeline")
        return modeling_pipeline


class ImagePipeline(_TfPipeline):

    def __init__(self, config: ImageConfig = ImageConfig):
        super().__init__(config)

        self.img_size = ()
        self.model_obj = Sequential

    def set_img_size_and_model_obj(self, img_size: Tuple[int, int], model_obj: Sequential):
        validate_variables(img_size, model_obj)

        self.img_size = img_size
        self.model_obj = model_obj
        self.logger.info("Successfully set img size and model obj")

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

        self.logger.info("Successfully loaded image processing pipeline")
        return processing_pipeline

    def get_model(self, model_obj=None):
        model_obj = model_obj or self.model_obj
        validate_variables(model_obj)

        model = model_obj(include_top=False, weights='imagenet', classes=len(self.config.DISEASES))
        model.trainable = False

        self.model = model

        self.logger.warning("Warning! get_model function in ImagePipeline returns unfitted model")
        return model

    def get_modeling_pipeline(self,
                              img_size: Tuple[int, int] = None, learning_rate: float = None,
                              metrics: List[str] = None, n_epochs: int = None,
                              train_dataset: Dataset = None, validation_dataset: Dataset = None):
        img_size = img_size or self.img_size
        learning_rate = learning_rate or self.config.LEARNING_RATE
        metrics = metrics or self.config.METRICS
        n_epochs = n_epochs or self.config.NUM_EPOCHS
        train_dataset = train_dataset or self.train_dataset
        validation_dataset = validation_dataset or self.validation_dataset

        processing_pipeline = self.get_processing_pipeline()
        model = self.get_model()

        validate_variables(img_size, learning_rate, metrics, n_epochs, processing_pipeline, model)

        modeling_pipeline = tf.keras.Sequential([tf.keras.Input(shape=img_size + (3,)),
                                                 processing_pipeline,
                                                 model,
                                                 tf.keras.layers.GlobalAveragePooling2D(),
                                                 tf.keras.layers.Dense(len(self.config.DISEASES), "softmax")
                                                 ])

        # TODO: Add weighted Adam
        model = self._compile_model(modeling_pipeline, learning_rate, metrics)
        modeling_pipeline = self._train_model(model, train_dataset, validation_dataset, n_epochs)

        self.modeling_pipeline = modeling_pipeline
        self.logger.info("Successfully loaded modeling pipeline")
        return modeling_pipeline


class TransformersModelingPipeline:
    def __init__(self, processing_pipeline, model):
        validate_variables(processing_pipeline, model)

        self.processing_pipeline = processing_pipeline
        self.model = model

    def __call__(self, dataset: Dataset, *args, **kwargs):
        validate_variables(dataset)

        predictions = self.predict(dataset)
        return predictions

    def evaluate(self, dataset: Dataset, batch_size: int = 4):
        validate_variables(dataset, batch_size)

        dataset_encoded = self.processing_pipeline(dataset)
        evaluations = self.model.evaluate(dataset_encoded.batch(batch_size))
        return evaluations

    def predict(self, dataset: Dataset, batch_size: int = 4):
        validate_variables(dataset, batch_size)

        dataset_encoded = self.processing_pipeline(dataset)
        predictions = self.model.predict(dataset_encoded.batch(batch_size))
        return predictions

    @classmethod
    def load_from_pretrained(cls, path: Path):
        model = TFDistilBertForSequenceClassification.from_pretrained(path)
        tokenizer = DistilBertTokenizerFast.from_pretrained(path)
        processing_pipeline = _TransformersProcessingPipeline(TextPipeline.encode_dataset, tokenizer)
        validate_variables(model, tokenizer, processing_pipeline)

        return cls(model=model,
                   processing_pipeline=processing_pipeline)


class _TransformersProcessingPipeline:
    def __init__(self, processing_function, tokenizer):
        validate_variables(processing_function, tokenizer)

        self.processing_function = processing_function
        self.tokenizer = tokenizer

    def __call__(self, dataset: Dataset, *args, **kwargs) -> Dataset:
        validate_variables(dataset)

        dataset_encoded = self.processing_function(dataset, self.tokenizer)
        return dataset_encoded


class TextPipeline(_SklearnPipeline, _TfPipeline):

    def __init__(self, config: TextConfig = TextConfig):
        _SklearnPipeline.__init__(self, config)
        _TfPipeline.__init__(self, config)
        self.tokenizer = None

    @staticmethod
    def encode_dataset(dataset, tokenizer) -> Dataset:
        validate_variables(dataset, tokenizer)

        text_to_encode = []
        labels = []
        for batch in dataset.as_numpy_iterator():
            text_batch = batch[0]
            labels_batch = batch[1]

            text_batch_decoded = [text.decode("utf-8") for text in text_batch]
            text_to_encode += text_batch_decoded
            labels += labels_batch.tolist()
        text_encodings = tokenizer(text_to_encode, truncation=True, padding=True)
        text_batch_dataset = tf.data.Dataset.from_tensor_slices((
            dict(text_encodings),
            labels))
        return text_batch_dataset

    def get_best_modeling_pipeline_type(self,
                                        transformer_modeling_pipeline: TransformersModelingPipeline = None,
                                        sklearn_modeling_pipeline: Pipeline = None,
                                        x_test: DataFrame = None,
                                        y_test: Series = None,
                                        test_dataset: Dataset = None)\
            -> Union[Pipeline, TransformersModelingPipeline]:
        x_test, y_test = self._set_dfs_test(x_test, y_test)
        test_dataset = test_dataset or self.test_dataset
        validate_variables(x_test, y_test, test_dataset)

        transformer_results_metric = (transformer_modeling_pipeline
                                      .evaluate(test_dataset, batch_size=self.config.BATCH_SIZE))[1]

        # TODO: Fix this to use specified metric
        sklearn_predictions = sklearn_modeling_pipeline.predict(x_test)
        sklearn_results_metric = accuracy_score(y_test, sklearn_predictions)

        if transformer_results_metric > sklearn_results_metric:
            modeling_pipeline = transformer_modeling_pipeline
        else:
            modeling_pipeline = sklearn_modeling_pipeline

        self.modeling_pipeline = modeling_pipeline

        self.logger.info("Successfully found best modeling pipeline type")
        return modeling_pipeline

    def get_model(self, use_sklearn=True, x_train=None, x_test=None, y_train=None, y_test=None):
        if use_sklearn:
            validate_variables(x_train, x_test, y_train, y_test)
            model = self._get_sklearn_model(x_train, x_test, y_train, y_test)
        else:
            model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                          num_labels=len(self.config.DISEASES))
        self.model = model

        self.logger.warning("Warning! get_model for transformers function in TextPipeline returns unfitted model")
        return model

    def get_processing_pipeline(self, use_sklearn=True):
        if use_sklearn:
            processing_pipeline = Pipeline([("Lemmatization, punctuation and stopwords removal",
                                             SpacyPreprocessor()),
                                            ("Tfidf Vectorizer", TfidfVectorizer(ngram_range=(1, 2)))])
        else:
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            self.tokenizer = tokenizer
            processing_pipeline = _TransformersProcessingPipeline(self.encode_dataset, tokenizer)

        self.processing_pipeline = processing_pipeline
        self.logger.info("Successfully loaded processing pipeline")
        return processing_pipeline

    def get_modeling_pipeline(self, use_sklearn=True,
                              x_train: DataFrame = None, x_test: DataFrame = None,
                              y_train: Series = None, y_test: Series = None,
                              train_dataset: Dataset = None, validation_dataset: Dataset = None,
                              learning_rate: float = None, metrics: List[str] = None, n_epochs: int = None):
        x_train, x_test, y_train, y_test = self._set_dfs(x_train, x_test, y_train, y_test)
        train_dataset = train_dataset or self.train_dataset
        validation_dataset = validation_dataset or self.validation_dataset
        validate_variables(x_train, x_test, y_train, y_test,
                           train_dataset, validation_dataset)

        processing_pipeline = self.get_processing_pipeline(use_sklearn)
        model = self.get_model(use_sklearn, x_train, x_test, y_train, y_test)

        validate_variables(x_train, x_test, y_train, y_test,
                           train_dataset, validation_dataset)

        if use_sklearn:
            modeling_pipeline = Pipeline([("Processing pipeline", processing_pipeline),
                                          ("Model", model)])
            modeling_pipeline.fit(x_train, y_train)
        else:
            model = self._compile_model(model, learning_rate, metrics)
            train_dataset, validation_dataset = (processing_pipeline(train_dataset),
                                                 processing_pipeline(validation_dataset))
            model = self._train_model(model, train_dataset, validation_dataset, n_epochs)
            modeling_pipeline = TransformersModelingPipeline(model=model,
                                                             processing_pipeline=processing_pipeline)

        self.modeling_pipeline = modeling_pipeline
        self.logger.info("Successfully loaded modeling pipeline")
        return modeling_pipeline
