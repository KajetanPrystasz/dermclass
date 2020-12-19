import numpy as np
import pandas as pd
import pytest

from dermclass_models2.validation import StructuredValidation, TextValidation, ValidationError, validate_variables


def test_validate_variables(structured_training_df):
    args_with_none = ["test", 1, None]
    args_pd = ["test2", 2, structured_training_df]

    with pytest.raises(TypeError):
        validate_variables(*args_with_none)
    # Check if doesn't raise an error
    validate_variables(*args_pd)


class TestStructuredValidation:

    def test_validate(self, testing_config, structured_training_df):
        testing_config.VARIABLE_ORDER = ['erythema', 'scaling', "age"]
        testing_config.NA_VALIDATION_VAR_DICT = {"NUMERIC_NA_NOT_ALLOWED": ["age"],
                                                 "ORDINAL_NA_NOT_ALLOWED": ["erythema", "scaling"],
                                                 "CATEGORICAL_NA_NOT_ALLOWED": []}
        validator = StructuredValidation(testing_config)
        df = structured_training_df[testing_config.VARIABLE_ORDER]

        df_nans = (df
                   .copy()
                   .assign(erythema=[1, np.NaN])
                   .assign(scaling=[2, 3])
                   .assign(age=[12, 25]))

        df_unexpected_col = df.copy().assign(extra_column=[1, 2])
        df_missing_col = df.copy().drop("erythema", axis=1)

        reordered_columns = structured_training_df[testing_config.VARIABLE_ORDER].columns.to_list()[::-1]
        df_reordered = (df
                        .copy()
                        .reindex(columns=reordered_columns))

        assert df_reordered.columns.to_list() == reordered_columns
        assert validator.validate(df_nans).shape[0] == 1
        with pytest.raises(ValidationError):
            validator.validate(df_unexpected_col)
        with pytest.raises(ValidationError):
            validator.validate(df_missing_col)


class TestTextValidation:

    def test_validate(self, testing_config):

        testing_config.VARIABLE_ORDER = ["text", "target"]
        df = pd.DataFrame(np.array([["Some testing text 1", 1],
                                    ["Some testing text 2", 2]]),
                          columns=["text", "target"])

        validator = TextValidation(testing_config)

        df_unexpected_col = df.copy().assign(extra_column=[1, 2])
        df_missing_col = df.copy().drop("text", axis=1)

        reordered_columns = df.columns.to_list()[::-1]
        df_reordered = (df
                        .copy()
                        .reindex(columns=reordered_columns))

        assert df_reordered.columns.to_list() == reordered_columns
        with pytest.raises(ValidationError):
            validator.validate(df_unexpected_col)
        with pytest.raises(ValidationError):
            validator.validate(df_missing_col)
