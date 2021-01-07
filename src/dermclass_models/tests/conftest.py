import pathlib
import pytest

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from dermclass_models.config import BaseConfig
import dermclass_models


class TestingConfig:
    PATIENCE = 3
    BATCH_SIZE = 1
    TUNING_FUNC_PARAMS = {"n_jobs": -1, "max_overfit": 0.9, "cv": 2, "n_trials": 1}
    DISEASES = ["test_disease0", "test_disease1", "test_disease2", "test_disease3", "test_disease4", "test_disease5"]
    SEED = 42
    METRICS = ["accuracy"]
    NUM_EPOCHS = 1

    PACKAGE_ROOT = pathlib.Path(dermclass_models.__file__).resolve().parent
    PICKLE_DIR = PACKAGE_ROOT / "pickles"


@pytest.fixture()
def testing_config():
    return TestingConfig


@pytest.fixture()
def structured_training_df():
    columns = ["erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon", "polygonal_papules",
               "follicular_papules", "oral_mucosal_involvement", "knee_and_elbow_involvement", "scalp_involvement",
               "family_history", "melanin_incontinence", "eosinophils_in_the_infiltrate", "pnl_infiltrate",
               "fibrosis_of_the_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis",
               "clubbing_of_the_rete_ridges", "elongation_of_the_rete_ridges",
               "thinning_of_the_suprapapillary_epidermis", "spongiform_pustule", "munro_microabcess",
               "focal_hypergranulosis", "disappearance_of_the_granular_layer",
               "vacuolisation_and_damage_of_basal_layer", "spongiosis", "saw_tooth_appearance_of_retes",
               "follicular_horn_plug", "perifollicular_parakeratosis", "inflammatory_monoluclear_inflitrate",
               "band_like_infiltrate", "age", "target"]
    values = [[2, 2, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 2, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 55, 2],
              [2, 2, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 2,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 53, 1]]
    df = pd.DataFrame(np.array(values), columns=columns)
    return df


@pytest.fixture()
def structured_set():
    df = pd.DataFrame(np.array([[2, 22, 1],
                                [2, 23, 1],
                                [1, 2, 0],
                                [0, 32, 2],
                                [2, 22, 1],
                                [2, 23, 1],
                                [1, 2, 0],
                                [0, 32, 2]
                                ]),
                      columns=["erythema", "age", "target"])
    x = df.drop("target", axis=1)
    y = df["target"]
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.25,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


@pytest.fixture()
def structured_text_set():
    df = pd.DataFrame(np.array([["Some sample text1", 1],
                                ["Some sample text2", 1],
                                ["Some sample text3", 0],
                                ["Some sample text4", 2],
                                ["Some sample text5", 1],
                                ["Some sample text6", 1],
                                ["Some sample text7", 0],
                                ["Some sample text8", 2],
                                ["Some sample text9", 0],
                                ["Some sample text10", 2]
                                ]),
                      columns=["text", "target"])
    x = df.drop("target", axis=1)
    y = df["target"]
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.25,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


@pytest.fixture()
def train_dataset():
    dataset = tf.data.Dataset.range(10).batch(2)
    return dataset


@pytest.fixture()
def text_train_dataset():
    train_dataset = (tf.keras.preprocessing.text_dataset_from_directory(directory=(BaseConfig.PACKAGE_ROOT /
                                                                                   ".." /
                                                                                   "tests" /
                                                                                   "test_text_dir"),
                                                                        batch_size=10))
    return train_dataset


# TODO: Add mocking for this fixture
@pytest.fixture()
def image_train_dataset():
    train_dataset = (tf.keras.preprocessing.image_dataset_from_directory(directory=(BaseConfig.PACKAGE_ROOT /
                                                                                    ".." /
                                                                                    "tests" /
                                                                                    "test_image_dir"),
                                                                         image_size=(456, 456),
                                                                         batch_size=10,
                                                                         seed=BaseConfig.SEED,
                                                                         shuffle=True))
    return train_dataset


def _xgboost_trial(trial) -> dict:
    params = {"subsample": trial.suggest_discrete_uniform("subsample", 0.1, 1, 0.1),
              "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.6, 1, 0.1)}
    return params


def _multinomial_nb_trial(trial) -> dict:
    params = {"alpha": trial.suggest_discrete_uniform("alpha", 0.1, 5, 0.1)}
    return params


@pytest.fixture()
def xgboost_trial():
    return _xgboost_trial


@pytest.fixture()
def multinomial_nb_trial():
    return _multinomial_nb_trial
