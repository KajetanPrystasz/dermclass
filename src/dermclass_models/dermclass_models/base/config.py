import pathlib
import logging
import sys
import dermclass_models


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

    trials_dict = {}

    tuning_func_params = {"n_jobs": -1, "max_overfit": 0.05, "cv": 4, "n_trials": 20}

    testing = False


class TestingConfig(BaseConfig):

    testing = True
    tuning_func_params = {"n_jobs": 1, "max_overfit": 1.0, "cv": 2, "n_trials": 1}


def get_console_handler() -> logging:
    """Use stream handler for logger tp log to stdout using proper format"""

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(BaseConfig.LOG_FORMAT)
    return console_handler
