import pathlib
import logging
import sys
import dermclass_text
from optuna import trial

PACKAGE_ROOT = pathlib.Path(dermclass_text.__file__).resolve().parent
PICKLE_DIR = PACKAGE_ROOT / "pickles"
DATA_PATH = PACKAGE_ROOT / "datasets"
PIPELINE_NAME = "text_pipeline"

SEED = 42

diseases = ["psoriasis", "lichen_planus"]

log_format = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s")

TUNING_FUNC_PARAMS = {"n_jobs": -1, "max_overfit": 0.2, "cv": 2, "n_trials": 10}

variables = {
    "TEXT" : ["text"],
    "TARGET": ["target"]
}

# TODO: Add more models for the future
def multinomial_nb_trial(trial : trial) -> dict:
    """This is a setup function for Optuna.study"""
    params = {"alpha": trial.suggest_discrete_uniform("alpha", 0.1, 5, 0.1)}
    return params

# trials_dict = {"XGBRFClassifier": xgboost_trial, "XGBClassifier": xgboost_trial}
trials_dict = {"MultinomialNB": multinomial_nb_trial}

def get_console_handler() -> logging:
    """Use stream handler for logger tp log to stdout using proper format"""

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    return console_handler
