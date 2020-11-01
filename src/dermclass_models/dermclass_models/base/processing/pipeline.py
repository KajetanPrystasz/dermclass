import abc
import logging

import pandas as pd
import numpy as np

from optuna import create_study, Trial
from optuna.samplers import TPESampler

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from dermclass_models.base.config import BaseConfig


class PpcPipeline:

    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.studies = {}

        self.final_model = None

    @abc.abstractmethod
    def fit_ppc_pipeline(self, x_train: pd.Dataframe = None) -> ColumnTransformer:
        pass

    def fit_data(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    # TODO: Add additional metrics
    @staticmethod
    def _hyper_param_optimization(trial, model_name: str, trial_func: Trial, max_overfit: float, cv: int,
                                  x_train: pd.DataFrame, x_test: pd.DataFrame,
                                  y_train: pd.DataFrame, y_test: pd.DataFrame):
        """Find optimal hyperparameters using bayesian hypeparameter tuning
        Function optimizes provided scoring metrics with the use of CV, taking to the account overfitting"""

        model_obj = eval(model_name)
        cv_score = np.mean(cross_val_score(model_obj(**trial_func(trial)),
                                           x_train,
                                           y_train,
                                           scoring="accuracy",
                                           cv=cv))

        model = model_obj(**trial_func(trial))
        model.fit(x_train, y_train)

        train_score = accuracy_score(y_train, model.predict(x_train))
        test_score = accuracy_score(y_test, model.predict(x_test))

        if abs(train_score - test_score) > max_overfit:
            output = 0
        else:
            output = cv_score
        return output

    # TODO: Add pruning
    def tune_hyperparameters(self, trials_dict,
                             x_train: pd.DataFrame = None, x_test: pd.DataFrame = None,
                             y_train: pd.DataFrame = None, y_test: pd.DataFrame = None,
                             max_overfit=0.05, cv=5, n_trials=20, n_jobs=-1):
        """Find best model and best hyperparameters from searched parameter space"""

        if any([x_train, x_test, y_train, y_test]) is not None:
            x_train = self.x_train
            x_test = self.x_test
            y_train = self.y_train
            y_test = self.y_test

        self.studies = {}
        for model_name, trial_func in trials_dict.items():
            self.logger.info(f"Finding hyperameters for {model_name}")
            sampler = TPESampler(seed=self.config.SEED)
            study = create_study(direction="maximize", study_name=model_name, sampler=sampler)
            study.optimize(lambda trial:
                           (self._hyper_param_optimization(trial, model_name, trial_func, max_overfit, cv,
                                                           x_train, x_test, y_train, y_test)),
                           n_trials=n_trials, n_jobs=n_jobs)

            self.studies[model_name] = study

        best_score, best_model, best_params = 0, self.config.DEFAULT_BEST_MODEL, {}
        for study in self.studies.values():
            if study.best_value > best_score:
                best_model = study.study_name
                best_score = study.best_value
                best_params = study.best_params
        self.logger.info(f"Best params found for: {best_model} with score: {best_score}")

        self.final_model = eval(best_model)(**best_params).fit(x_train, y_train)
        self.logger.info("Successfully tuned hyperparameters")
        return self.final_model
