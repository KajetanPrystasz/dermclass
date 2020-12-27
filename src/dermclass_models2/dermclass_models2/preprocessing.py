from typing import Tuple, Union, List
import abc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import cv2

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin, BaseEstimator

from dermclass_models2.config import StructuredConfig, ImageConfig, TextConfig
from dermclass_models2.validation import validate_variables

DataFrame = pd.DataFrame
Series = pd.Series
Dataset = tf.data.Dataset
Sequential = tf.keras.models.Sequential


class _SklearnPreprocessors(abc.ABC):

    def __init__(self, config):
        """
        An abstract class for  for preprocessing data with sklearn
        :param config: Config object for the class
        """
        validate_variables(config)
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.df = pd.DataFrame

        self.x = pd.DataFrame
        self.y = pd.Series

        self.x_train = pd.DataFrame
        self.x_test = pd.DataFrame
        self.y_train = pd.Series
        self.y_test = pd.Series

    @abc.abstractmethod
    def _load_structured_data(self, path: Path) -> DataFrame:
        """
        Abstract method for loading structured data
        :param path: Path to data directory
        :return: Returns a pandas DataFrame with loaded data
        """
        return DataFrame()

    def _split_target_structured(self, df: DataFrame = None, target_col: str = None)\
            -> Tuple[DataFrame, Series]:
        """
        Utility function to split target column from pandas DataFrame
        :param df: A pandas DataFrame to split the target column from
        :param target_col: Name of column with target data
        :return: Returns a tuple of pandas DataFrame and pandas Series with target data
        """
        if isinstance(df, DataFrame):
            df = df
        elif df is None:
            df = self.df
        target_col = target_col or self.config.TARGET
        validate_variables(df, target_col)

        x = df.drop(target_col, 1)
        y = df[target_col]

        self.x = x
        self.y = y
        self.logger.info("Successfully splat the target")
        return x, y

    def _split_train_test_structured(self,
                                     x: DataFrame = None, y: Series = None,
                                     test_size: float = 0.2, random_state: int = 42)\
            -> Tuple[DataFrame, DataFrame, Series, Series]:
        if isinstance(x, pd.DataFrame):
            x = x
        elif x is None:
            x = self.x
        if isinstance(y, pd.DataFrame):
            y = y
        elif y is None:
            y = self.y
        test_size = test_size or self.config.TEST_SIZE
        random_state = random_state or self.config.SEED
        validate_variables(x, y, test_size, random_state)

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=test_size,
                                                            random_state=random_state)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.logger.info("Successfully splat train and test data")
        return x_train, x_test, y_train, y_test

    def _load_data_structured(self, df: DataFrame = None) -> Tuple[DataFrame, DataFrame, Series, Series]:
        x, y = self._split_target_structured(df)
        x_train, x_test, y_train, y_test = self._split_train_test_structured(x, y)

        self.logger.info("Successfully loaded the data")
        return x_train, x_test, y_train, y_test


class _TfPreprocessors(abc.ABC):

    def __init__(self, config):
        """
        An abstract class for  for preprocessing data with tensorflow
        :param config: Config object for the class
        """
        validate_variables(config)
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.train_dataset = Dataset
        self.validation_dataset = Dataset
        self.test_dataset = Dataset

        self.prefetch = True

    def _split_train_test_tf(self, train_dataset: Dataset = None, validation_dataset: Dataset = None):
        """
        Utility function to split test data from validation dataset and set datasets to prefetch mode. Used to reduce
        neural net bottleneck duing data loading stage
        :param train_dataset: A train dataset
        :param validation_dataset: A validation dataset to be split into validation and test datasets of equal size
        :return: Returns a prefetched train, validation, testdatasets
        """
        train_dataset = train_dataset or self.train_dataset
        validation_dataset = validation_dataset or self.validation_dataset
        validate_variables(train_dataset, validation_dataset)

        validation_batches = tf.data.experimental.cardinality(validation_dataset)
        test_dataset = validation_dataset.take(validation_batches // 2)
        validation_dataset = validation_dataset.skip(validation_batches // 2)

        if self.prefetch:
            train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        self.logger.info(f"Successfully prefetched train, test and validation datasets")
        self.logger.info(f'Number of train batches: {tf.data.experimental.cardinality(train_dataset)}\
        Number of validation batches: {tf.data.experimental.cardinality(validation_dataset)}\
        Number of test batches: {tf.data.experimental.cardinality(test_dataset)}')

        return train_dataset, validation_dataset, test_dataset


class StructuredPreprocessor(_SklearnPreprocessors):

    def __init__(self, config: StructuredConfig = StructuredConfig):
        """
        A class for preprocessing structured data
        :param config: Config object for the class
        """
        validate_variables(config)
        super().__init__(config)

    def _load_structured_data(self, path: Path = None) -> DataFrame:
        """
        Utility function to loaod structured data from the csv
        :param path: Path to data file
        :return: Returns a pandas DataFrame with data loaded
        """
        path = path or self.config.DATA_PATH
        validate_variables(path)

        df = pd.read_csv(path)
        self.df = df
        self.logger.info("Successfully loaded data from csv")
        return df

    def load_data(self, path: Path = None) -> Tuple[DataFrame, DataFrame, Series, Series]:
        """
        Function to load structured data using sklearn
        :param path: Path to data directory
        :return: Returns a tuple with x_train, x_test, y_train, y_test data
        """
        validate_variables(path)

        df = self._load_structured_data(path)
        x_train, x_test, y_train, y_test = self._load_data_structured(df)
        return x_train, x_test, y_train, y_test


class ImagePreprocessors(_TfPreprocessors):

    def __init__(self, config: ImageConfig = ImageConfig):
        """
        A class for preprocessing image data
        :param config: Config object for the class
        """
        super().__init__(config)

        self.config = config
        self.logger = logging.getLogger(__name__)

        self.model = None

        self.img_size = ()
        self.img_shape = ()

    def _get_avg_img_size(self, path: Path = None) -> Tuple[int, int]:
        """
        Utility function to get average image size from your data, necessary to choose proper EfficientNet version
        :param path: Path to data files
        :return: Returns a tuple with mean image size from provided image data
        """
        path = path or self.config.DATA_PATH
        validate_variables(path)

        height_list = []
        width_list = []
        for subclass_dir in path.iterdir():
            for img_path in subclass_dir.iterdir():
                img = cv2.imread(str(img_path))
                height, width, _ = img.shape
                height_list.append(height)
                width_list.append(width)
        mean_height = int(sum(height_list) / len(height_list))
        mean_width = int(sum(width_list) / len(width_list))

        self.img_size = (mean_height, mean_width)

        self.logger.info(f"Mean height is: {mean_height}, mean width is: {mean_width}")
        return self.img_size

    def _get_efficientnet_and_size(self, img_size: Tuple[int, int] = None) -> Tuple[Tuple[int, int], Sequential]:
        """
        Utility function to get proper type of EfficientNet, the version is chosen based on the mean image size
        More on: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
        :param img_size: Mean input size of image files
        :return: Returns a tuple of image_size after changing to right static value and EfficientNet object
        """
        img_size = img_size or self.img_size
        validate_variables(img_size)

        img_size = (img_size[0] + img_size[1]) / 2

        if 564 < img_size:
            img_size = (600, 600)
            model = tf.keras.applications.EfficientNetB7
        elif 492 < img_size <= 564:
            img_size = (528, 528)
            model = tf.keras.applications.EfficientNetB6
        else:
            img_size = (456, 456)
            model = tf.keras.applications.EfficientNetB5

        self.img_size = img_size
        self.model = model

        self.logger.info(f"Chosen model is {model} with img_size {img_size}")
        return img_size, model

    def _load_dataset(self, batch_size: int = None, data_path: Path = None, img_size: Tuple[int, int] = None)\
            -> Tuple[Dataset, Dataset]:
        """
        Utility function to load dataset from provided path
        :param batch_size: A batch size for the datasets
        :param data_path: Path to data directory. Files should be organized for tensorflow's image_dataset_from_directory
        :param img_size: Image size for loading the image data
        :return: Returns a tuple with train and validation datasets
        """
        batch_size = batch_size or self.config.BATCH_SIZE
        data_path = data_path or self.config.DATA_PATH
        img_size = img_size or self.img_size
        validate_variables(batch_size, data_path, img_size)

        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(directory=data_path,
                                                                            image_size=img_size,
                                                                            validation_split=self.config.TEST_SIZE,
                                                                            batch_size=batch_size,
                                                                            subset="training",
                                                                            seed=self.config.SEED,
                                                                            shuffle=True)

        validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(directory=data_path,
                                                                                 image_size=img_size,
                                                                                 validation_split=self.config.TEST_SIZE,
                                                                                 batch_size=batch_size,
                                                                                 subset="validation",
                                                                                 seed=self.config.SEED,
                                                                                 shuffle=True)

        self.train_dataset = validation_dataset
        self.validation_dataset = validation_dataset

        self.logger.info(f"Successfully loaded train and validation datasets ")
        return train_dataset, validation_dataset

    def load_data(self, path: Path = None) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Function to load image data using tensorflow
        :param path: Path to data directory
        :return: Returns train, validation and test datasets
        """
        path = path or self.config.DATA_PATH
        validate_variables(path)

        img_size = self._get_avg_img_size(path)
        img_size, _ = self._get_efficientnet_and_size(img_size)
        train_dataset, validation_dataset = self._load_dataset(img_size=img_size, data_path=path)
        train_dataset, validation_dataset, test_dataset = self._split_train_test_tf(train_dataset,
                                                                                    validation_dataset)

        return train_dataset, validation_dataset, test_dataset


class TextPreprocessors(_SklearnPreprocessors, _TfPreprocessors):

    def __init__(self, config: TextConfig = TextConfig):
        """
        A class for preprocessing text data
        :param config: Config object for the class
        """
        _SklearnPreprocessors.__init__(self, config)
        _TfPreprocessors.__init__(self, config)

        self.config = config
        self.logger = logging.getLogger(__name__)

    def _load_class_from_dir(self, path: Path) -> DataFrame:
        """
        Utility function for csvs from a directory to one pandas data frame
        :param path: A path to the directory
        :return: Returns pandas DataFrame with csv's loaded from one class directory
        """
        validate_variables(path)
        class_name = path.name

        df = pd.DataFrame()
        index_num = [0]
        for child in path.iterdir():
            with open(str(child), "r", encoding='utf-8') as file:
                text = file.read().replace("\n", " ")
                file_df = pd.DataFrame({"target": class_name,
                                        "text": text},
                                       index=index_num)
                index_num[0] = index_num[0] + 1
                df = pd.concat([df, file_df], axis=0)

        self.logger.info(f"Successfully loaded class {class_name}")
        return df

    def _load_structured_data(self, path: Path = None) -> DataFrame:
        """
        Utility function for loading all classes in provided path
        :param path: A path to structured data directory
        :return: Returns a pandas DataFrame with all classes laoded
        """
        path = path or self.config.DATA_PATH
        validate_variables(path)
        df = pd.DataFrame()

        for class_dir in path.iterdir():
            class_df = self._load_class_from_dir(class_dir)
            df = pd.concat([df, class_df])
        df = df.reset_index().drop("index", axis=1)

        self.df = df
        self.logger.info("Successfully loaded the data from file")
        return df

    def _load_dataset(self, batch_size: int = None, data_path: Path = None) -> Tuple[Dataset, Dataset]:
        """
        Utility function for loading tensorflow dataset from provided directory. The function splits train and validation
        :param batch_size: A batch size for the datasets
        :param data_path: Path to data directory. Files should be organized for tensorflow's text_dataset_from_directory
        :return: Returns a tuple with train and validation datasets
        """
        batch_size = batch_size or self.config.BATCH_SIZE
        data_path = data_path or self.config.DATA_PATH
        validate_variables(batch_size, data_path)

        train_dataset = tf.keras.preprocessing.text_dataset_from_directory(directory=data_path,
                                                                           validation_split=self.config.TEST_SIZE,
                                                                           batch_size=batch_size,
                                                                           subset="training",
                                                                           seed=self.config.SEED,
                                                                           shuffle=True)

        validation_dataset = tf.keras.preprocessing.text_dataset_from_directory(directory=data_path,
                                                                                validation_split=self.config.TEST_SIZE,
                                                                                batch_size=batch_size,
                                                                                subset="validation",
                                                                                seed=self.config.SEED,
                                                                                shuffle=True)

        self.train_dataset = validation_dataset
        self.validation_dataset = validation_dataset

        self.logger.info(f"Successfully loaded train and validation datasets ")
        return train_dataset, validation_dataset

    def load_data(self, path: Path = None, get_datasets: bool = False)\
            -> Union[Tuple[Dataset, Dataset, Dataset],
                     Tuple[DataFrame, DataFrame, Series, Series]]:
        """
        Function to load text data using either sklearn or tensorflow
        :param path: Path to data directory
        :param get_datasets: A boolean indicating if function should return datasets (as for tensorflow) or pandas DataFrames
        :return: Returns train, validation, test datasets or x_train, x_test, y_train, y_test tuples
        """
        path = path or self.config.DATA_PATH
        validate_variables(path)

        if get_datasets:
            train_dataset, validation_dataset = self._load_dataset(data_path=path)
            train_dataset, validation_dataset, test_dataset = self._split_train_test_tf(train_dataset,
                                                                                        validation_dataset)
            return train_dataset, validation_dataset, test_dataset
        else:
            df = self._load_structured_data(path)
            x_train, x_test, y_train, y_test = self._load_data_structured(df)
            return x_train, x_test, y_train, y_test


class CastTypesTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, categorical_variables: List[str], ordinal_variables: List[str],
                 numeric_variables: List[str]):
        """
        Custom sklearn transformer to change types of variables to be used with sklearn ColumnTransformer
        :param categorical_variables: A list of variables to be cast to categorical type
        :param ordinal_variables: A list of variables to be cast to integer type
        :param numeric_variables: A list of variables to be cast to float32 type
        """
        self.categorical_variables = categorical_variables
        self.ordinal_variables = ordinal_variables
        self.numeric_variables = numeric_variables

        self.x_cat = None
        self.x_ord = None
        self.x_num = None

        self.x = None
        self.y = None

    def fit(self, x, y=None):
        """
        Function to fit data
        :param x: X param to fit data
        :param y: Optional y parameter
        """
        self.x = x
        self.y = y
        return self

    def transform(self, x=None, y=None):
        """
        Funtion to transform fitted data
        :param x: Optional parameter to transform data, if x is None it uses fitted values
        :param y: Optional y parameter
        :return: Return pandas DataFrame with changed types
        """
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


class SpacyPreprocessor(TransformerMixin, BaseEstimator):

    def __init__(self):
        """
        A preprocessor to be used with sklearn ColumnTransformer which uses spacy to do lemmatization, tokenization and
        stop words removal
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.doc = None
        self.x = None

    def fit(self, x, y=None):
        """
        Function to fit data
        :param x: X param to fit data
        :param y: Optional y parameter
        """
        self.x = x
        return self

    def transform(self, x=None, y=None):
        """
        Funtion to perform lemmatization, tokenization and stop words removal
        :param x: Optional parameter to transform data, if x is None it uses fitted values
        :param y: Optional y parameter
        :return: Return pandas DataFrame with data ready for classification or futher tranformations
        """
        if x is None:
            x = self.x

        # TODO: Change to apply on df instead of iterating over df rows
        array = np.array([])
        for row in x["text"]:
            tokens = self.nlp(row)
            output_text = " ".join([token.lemma_ for token in tokens if not (token.is_punct or token.is_stop)])
            array = np.append(array, output_text)
        return array
