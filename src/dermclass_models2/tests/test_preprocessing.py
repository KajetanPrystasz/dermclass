from pathlib import Path

import pandas as pd
import tensorflow as tf
import numpy as np

from dermclass_models2.preprocessing import (StructuredPreprocessor,
                                             TextPreprocessors,
                                             ImagePreprocessors,
                                             CastTypesTransformer,
                                             SpacyPreprocessor)


def _assert_structured(x_train, x_test, y_train, y_test):
    assert "target" not in x_train.columns + x_test.columns
    assert y_train.name == "target" and y_test.name == "target"
    assert x_train.shape == (3, 34)
    assert x_test.shape == (1, 34)
    assert y_train.shape == (3,)
    assert y_test.shape == (1,)


def _assert_tf(train_dataset, validation_dataset, test_dataset):
    train_batches = tf.data.experimental.cardinality(train_dataset)
    validation_batches = tf.data.experimental.cardinality(validation_dataset)
    test_batches = tf.data.experimental.cardinality(test_dataset)

    assert train_batches == 5
    assert validation_batches == 3
    assert test_batches == 2


class TestStructuredPreprocessor:

    def test_load_data(self, testing_config, structured_training_df, monkeypatch):
        testing_config.TARGET = "target"
        preprocessor = StructuredPreprocessor(testing_config)

        def mock__load_structured_data(path):
            df = pd.concat([structured_training_df, structured_training_df])
            return df

        monkeypatch.setattr(preprocessor, "_load_structured_data", mock__load_structured_data)
        x_train, x_test, y_train, y_test = preprocessor.load_data(path=Path("test_path"))
        _assert_structured(x_train, x_test, y_train, y_test)


class TestImagePreprocessors:

    def test_load_data(self, testing_config, structured_training_df, train_dataset, monkeypatch):
        preprocessor = ImagePreprocessors(testing_config)

        monkeypatch.setattr(preprocessor, "_get_avg_img_size", lambda path: (255, 255))
        monkeypatch.setattr(preprocessor, "_load_dataset", lambda img_size, data_path: (train_dataset, train_dataset))

        train_dataset, validation_dataset, test_dataset = preprocessor.load_data(path=Path("test_path"))

        _assert_tf(train_dataset, validation_dataset, test_dataset)
        # TODO: Add check for size


class TestTextPreprocessors:

    def test_load_data(self, testing_config, structured_training_df, train_dataset, monkeypatch):
        testing_config.TARGET = "target"
        preprocessor = TextPreprocessors(testing_config)

        def mock__load_structured_data(path):
            df = pd.concat([structured_training_df, structured_training_df])
            return df

        monkeypatch.setattr(preprocessor, "_load_structured_data", mock__load_structured_data)
        x_train, x_test, y_train, y_test = preprocessor.load_data(path=Path("test_path"))

        monkeypatch.setattr(preprocessor, "_load_dataset", lambda data_path: (train_dataset, train_dataset))
        train_dataset, validation_dataset, test_dataset = preprocessor.load_data(path=Path("test_path"),
                                                                                 get_datasets=True)

        _assert_structured(x_train, x_test, y_train, y_test)
        _assert_tf(train_dataset, validation_dataset, test_dataset)


class TestCastTypesTransformer:

    def test_fit(self, structured_training_df):
        df = structured_training_df[['erythema', 'scaling', "age"]]
        transformer = CastTypesTransformer(categorical_variables=[],
                                           ordinal_variables=['erythema', 'scaling'],
                                           numeric_variables=["age"])
        transformer.fit(df)

        assert transformer.x.equals(df)

    def test_transform(self, structured_training_df):
        df = structured_training_df[['erythema', 'scaling', "age"]]
        transformer = CastTypesTransformer(categorical_variables=[],
                                           ordinal_variables=['erythema', 'scaling'],
                                           numeric_variables=["age"])

        transformer.fit(df)
        df_transformed = transformer.transform()

        assert df_transformed["erythema"].dtype == "int"
        assert df_transformed["scaling"].dtype == "int"
        assert df_transformed["age"].dtype == "float32"


class TestSpacyPreprocessor:

    def test_fit(self):

        df = pd.DataFrame(np.array([["did good, words and other stuff", 2]]),
                          columns=["text", "target"])

        transformer = SpacyPreprocessor()
        transformer.fit(df)

        assert transformer.x.equals(df)

    def test_transform(self):
        df = pd.DataFrame(np.array([["did good, words and other stuff", 2]]),
                          columns=["text", "target"])

        transformer = SpacyPreprocessor()
        transformer.fit(df)

        df_transformed = transformer.transform()

        assert df_transformed[0] == 'good word stuff'
