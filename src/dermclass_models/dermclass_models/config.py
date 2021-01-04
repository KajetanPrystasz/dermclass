import pathlib
import logging
import sys
import abc

from optuna import trial

import dermclass_models

import tensorflow as tf


# TODO: Add more models for the future
def xgboost_trial(trial : trial) -> dict:
    """
    A xgboost trial function te be run inside optuna optimization function
    :param trial: trial object from optuna
    :return: Returns dictionary with trial params
    """
    params = {"subsample": trial.suggest_discrete_uniform("subsample", 0.1, 1, 0.1),
              "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.6, 1, 0.1),
              "colsample_bylevel": trial.suggest_discrete_uniform("colsample_bylevel", 0.6, 1, 0.1),
              "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
              "max_depth": trial.suggest_int("max_depth", 3, 12),
              "max_delta_step": trial.suggest_discrete_uniform("max_delta_step", 0.1, 10, 0.1),
              "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
              "n_estimators": trial.suggest_int("n_estimators", 100, 500),
              "gamma": trial.suggest_discrete_uniform("gamma", 0.1, 30, 0.1)}
    return params


def multinomial_nb_trial(trial: trial) -> dict:
    """
    A multinomial naive bayes trial function te be run inside optuna optimization function
    :param trial: trial object from optuna
    :return: Returns dictionary with trial params
    """
    params = {"alpha": trial.suggest_discrete_uniform("alpha", 0.1, 5, 0.1)}
    return params


class BaseConfig:

    PACKAGE_ROOT = pathlib.Path(dermclass_models.__file__).resolve().parent
    PICKLE_DIR = PACKAGE_ROOT / "pickles"

    LOG_FORMAT = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s")
    TEST_SIZE = 0.2

    SEED = 42
    TARGET = "target"

    PIPELINE_TYPE = ""

    VARIABLE_ORDER = []

    DEFAULT_BEST_MODEL = "XGBClassifier"
    DEFAULT_BEST_DL_MODEL = "TFDistilBertForSequenceClassification"

    TRIALS_DICT = {}

    TUNING_FUNC_PARAMS = {"n_jobs": -1, "max_overfit": 0.05, "cv": 3, "n_trials": 10}


class StructuredConfig(BaseConfig):
    LABEL_MAPPING = {
        1: "psoriasis",
        2: "seboreic dermatitis",
        3: "lichen planus",
        4: "pityriasis rosea",
        5: "cronic dermatitis",
        6: "pityriasis rubra pilaris"
    }

    PIPELINE_TYPE = "structured_pipeline"

    VARIABLE_ORDER = [
        'erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon','polygonal_papules',
        'follicular_papules', 'oral_mucosal_involvement', 'knee_and_elbow_involvement', 'scalp_involvement',
        'family_history', 'melanin_incontinence', 'eosinophils_in_the_infiltrate', 'pnl_infiltrate',
        'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis',
        'clubbing_of_the_rete_ridges', 'elongation_of_the_rete_ridges', 'thinning_of_the_suprapapillary_epidermis',
        'spongiform_pustule', 'munro_microabcess', 'focal_hypergranulosis', 'disappearance_of_the_granular_layer',
        'vacuolisation_and_damage_of_basal_layer', 'spongiosis', 'saw_tooth_appearance_of_retes',
        'follicular_horn_plug', 'perifollicular_parakeratosis', 'inflammatory_monoluclear_inflitrate',
        'band_like_infiltrate', "age", "target"]

    NA_VALIDATION_VAR_DICT = {
        "NUMERIC_NA_ALLOWED": ["age"],
        "NUMERIC_NA_NOT_ALLOWED": [],

        "CATEGORICAL_NA_ALLOWED": [],
        "CATEGORICAL_NA_NOT_ALLOWED": [],

        "ORDINAL_NA_ALLOWED": [],
        "ORDINAL_NA_NOT_ALLOWED": [var for var in VARIABLE_ORDER if var not in ["age", "target"]]
    }

    DEFAULT_BEST_MODEL = "XGBClassifier"

    DATA_PATH = BaseConfig.PACKAGE_ROOT / "datasets" / "structured" / "dermatology_dataset.csv"

    TRIALS_DICT = {"XGBClassifier": xgboost_trial}


class _TfConfig(abc.ABC):

    BATCH_SIZE = 2
    NUM_EPOCHS = 5
    METRICS = ["accuracy"]
    LEARNING_RATE = 0.0001

    DISEASES = ["psoriasis", "lichen_planus", "pityriasis_rosea"]

    GPU_CONFIG = tf.compat.v1.ConfigProto()
    #GPU_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.5
    GPU_CONFIG.gpu_options.allow_growth = True

    PATIENCE = 3


class ImageConfig(BaseConfig, _TfConfig):

    PIPELINE_TYPE = "image_pipeline"

    VARIABLE_ORDER = ["image", "target"]

    DATA_PATH = BaseConfig.PACKAGE_ROOT / "datasets" / "image"

    IMG_HEIGHT = 255
    IMG_WIDTH = 255


class TextConfig(BaseConfig, _TfConfig):

    PIPELINE_TYPE = "text_pipeline"

    VARIABLE_ORDER = ["text", "target"]

    DEFAULT_BEST_MODEL = "MultinomialNB"

    DATA_PATH = BaseConfig.PACKAGE_ROOT / "datasets" / "text"

    TRIALS_DICT = {"MultinomialNB": multinomial_nb_trial}

    TUNING_FUNC_PARAMS = {"n_jobs": -1, "max_overfit": 0.2, "cv": 2, "n_trials": 10}


def get_console_handler() -> logging:
    """Use stream handler for logger tp log to stdout using proper format"""

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(BaseConfig.LOG_FORMAT)
    return console_handler
