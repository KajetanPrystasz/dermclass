import numpy as np
import pandas as pd
import pytest

from dermclass_models2.validation import StructuredValidation, TextValidation, ValidationError, validate_variables


def test_validate_variables(structured_training_set):
    args_with_none = ["test", 1, None]
    args_pd = ["test2", 2, structured_training_set]

    with pytest.raises(TypeError):
        validate_variables(*args_with_none)
    # Check if doesn't raise an error
    validate_variables(*args_pd)


class Base_SklearnValidation:

    def base__reorder_df(self, testing_config, structured_training_set, validation_class):
        validator = validation_class(testing_config)

        reordered_columns = structured_training_set.columns.to_list()
        reordered_columns.reverse()
        df_reordered = validator._reorder_df(structured_training_set, reordered_columns)

        assert df_reordered.columns.to_list() == reordered_columns

    def base__validate_columns(self, testing_config, structured_training_set, validation_class):
        validator = validation_class(testing_config)

        df = structured_training_set.copy()
        df["extra_column"] = [1, 2]
        df.drop("age", axis=1, inplace=True)

        validated_df = validator._validate_columns(structured_training_set, structured_training_set.columns.tolist())
        assert not validated_df.empty
        with pytest.raises(ValidationError):
            validator._validate_columns(df, structured_training_set.columns.tolist())

    def test_validate(self):
        pass


class TestStructuredValidation(Base_SklearnValidation):

    def test__drop_unexpected_na(self, testing_config, structured_training_set):

        testing_config.NA_VALIDATION_VAR_DICT = {"NUMERIC_NA_NOT_ALLOWED": ["age"],
                                                 "ORDINAL_NA_NOT_ALLOWED": ["erythema"],
                                                 "CATEGORICAL_NA_NOT_ALLOWED": ["erythema"]
                                                 }

        df = (structured_training_set[["age", "erythema"]]
              .assign(age=[12, np.NaN])
              .assign(erythema=[1, 1]))

        validator = StructuredValidation(testing_config)
        df_validated = validator._drop_unexpected_na(df)
        print(df_validated)
        assert df_validated.shape[0] == 1

    def test__reorder_df(self, testing_config, structured_training_set):
        super().base__reorder_df(testing_config, structured_training_set, validation_class=StructuredValidation)

    def test__validate_columns(self, testing_config, structured_training_set):
        super().base__validate_columns(testing_config, structured_training_set, validation_class=StructuredValidation)

    def test_validate(self):
        pass


class TestTextValidation(Base_SklearnValidation):

    def test__reorder_df(self, testing_config, structured_training_set):
        super().base__reorder_df(testing_config, structured_training_set, validation_class=TextValidation)

    def test__validate_columns(self, testing_config, structured_training_set):
        super().base__validate_columns(testing_config, structured_training_set, validation_class=TextValidation)

    def test_validate(self):
        pass
