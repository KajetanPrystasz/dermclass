from optuna import trial

from dermclass_models.base.config import BaseConfig, TestingConfig


# TODO: Add more models for the future
def multinomial_nb_trial(trial: trial) -> dict:
    """This is a setup function for Optuna.study"""
    params = {"alpha": trial.suggest_discrete_uniform("alpha", 0.1, 5, 0.1)}
    return params


class TextConfig(BaseConfig):

    PIPELINE_TYPE = "text_pipeline"

    VARIABLE_ORDER = ["text", "target"]

    DEFAULT_BEST_MODEL = "MultinomialNB"

    DATA_PATH = BaseConfig.PACKAGE_ROOT/"text"/"datasets"

    DISEASES = ["psoriasis", "lichen_planus"]

    trials_dict = {"MultinomialNB": multinomial_nb_trial}

    tuning_func_params = {"n_jobs": -1, "max_overfit": 0.2, "cv": 2, "n_trials": 10}


class TestingTextConfig(TestingConfig, TextConfig):
    pass
