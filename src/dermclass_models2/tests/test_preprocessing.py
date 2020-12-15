from dermclass_models2.preprocessing import (StructuredPreprocessor,
                                             TextPreprocessors,
                                             ImagePreprocessors)
import pandas as pd
import tensorflow as tf


class BaseSklearnPreprocessors:

    def base__split_target_structured(self, testing_config, structured_training_set, preprocessor_class):
        preprocessors = preprocessor_class(testing_config)

        x, y = preprocessors._split_target_structured(structured_training_set, "target")
        assert "target" not in x.columns
        assert y.name == "target"

    def base__split_train_test_structured(self, testing_config, structured_training_set, preprocessor_class):
        preprocessors = preprocessor_class(testing_config)

        df = structured_training_set.copy()
        df = pd.concat([df, df])
        x = df.drop("target", axis=1)
        y = df["target"]

        x_train, x_test, y_train, y_test = preprocessors._split_train_test_structured(x, y, test_size=0.25)

        assert x_train.shape == (3, 34)
        assert x_test.shape == (1, 34)
        assert y_train.shape == (3,)
        assert y_test.shape == (1,)

    def base__load_data_structured(self):
        pass


class BaseTfPreprocessors:

    def base_split_train_test_tf(self, testing_config, train_dataset, preprocessor_class):
        preprocessor = preprocessor_class(testing_config)

        validation_dataset = train_dataset

        train_dataset, validation_dataset, test_dataset = (preprocessor
                                                           ._split_train_test_tf(train_dataset, validation_dataset))

        train_batches = tf.data.experimental.cardinality(train_dataset)
        validation_batches = tf.data.experimental.cardinality(validation_dataset)
        test_batches = tf.data.experimental.cardinality(test_dataset)

        assert train_batches == 10
        assert validation_batches == 5
        assert test_batches == 5


class TestStructuredPreprocessor(BaseSklearnPreprocessors):

    def test__validate_columns(self, testing_config, structured_training_set):
        super().base__split_target_structured(testing_config,
                                              structured_training_set,
                                              preprocessor_class=StructuredPreprocessor)

    def test__split_train_test_structured(self, testing_config, structured_training_set):
        super().base__split_train_test_structured(testing_config,
                                                  structured_training_set,
                                                  preprocessor_class=StructuredPreprocessor)

    def test__load_data_structured(self):
        super().base__load_data_structured()

    def test__load_structured_data(self):
        pass

    def test_load_data(self):
        pass


class TestImagePreprocessors(BaseTfPreprocessors):

    def test__split_train_test_tf(self, testing_config, train_dataset):
        super().base_split_train_test_tf(testing_config,
                                         train_dataset,
                                         preprocessor_class=ImagePreprocessors)

    def test__get_avg_img_size(self):
        pass

    def test__get_efficientnet_and_size(self):
        pass

    def test__load_dataset(self):
        pass

    def test_load_data(self):
        pass


class TestTextPreprocessors(BaseSklearnPreprocessors, BaseTfPreprocessors):

    def test__validate_columns(self, testing_config, structured_training_set):
        super().base__split_target_structured(testing_config,
                                             structured_training_set,
                                             preprocessor_class=TextPreprocessors)

    def test__split_train_test_structured(self, testing_config, structured_training_set):
        super().base__split_train_test_structured(testing_config,
                                                  structured_training_set,
                                                  preprocessor_class=TextPreprocessors)

    def test__split_train_test_tf(self, testing_config, train_dataset):
        super().base_split_train_test_tf(testing_config,
                                         train_dataset,
                                         preprocessor_class=TextPreprocessors)

    def test__load_data_structured(self):
        super().base__load_data_structured()

    def test__load_class_from_dir(self):
        pass

    def test__load_structured_data(self):
        pass

    def test__load_dataset(self):
        pass

    def test_load_data(self):
        pass


class TestCastTypesTransformer:

    def test_fit(self):
        pass

    def test_transform(self):
        pass


class TestSpacyPreprocessor:

    def test_fit(self):
        pass

    def test_transform(self):
        pass
