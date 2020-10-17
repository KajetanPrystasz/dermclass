import logging
import numpy as np
from optuna import create_study
from optuna.samplers import TPESampler

from xgboost import XGBClassifier, XGBRFClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from dermclass_structured.processing.preprocessors import CastTypesTransformer
from dermclass_structured import config

_logger = logging.getLogger(__name__)
variables = config.variables
categorical_variables = variables["CATEGORICAL_NA_ALLOWED"] + variables["CATEGORICAL_NA_NOT_ALLOWED"]
ordinal_variables = variables["ORDINAL_NA_ALLOWED"] + variables["ORDINAL_NA_NOT_ALLOWED"]
numeric_variables = variables["NUMERIC_NA_ALLOWED"] + variables["NUMERIC_NA_NOT_ALLOWED"]
all_variables = categorical_variables + ordinal_variables + numeric_variables


def fit_ppc_pipeline(x_train):
    """Fit provided x_train data to preprocessing data"""

    ppc_pipeline = ColumnTransformer(transformers=[
        ("Cast dtypes", CastTypesTransformer(categorical_variables=categorical_variables,
                                             ordinal_variables=ordinal_variables,
                                             numeric_variables=numeric_variables), all_variables),

        ("Fill_na_categorical", SimpleImputer(strategy='most_frequent'), variables["CATEGORICAL_NA_ALLOWED"]),
        ("Fill_na_ordinal", SimpleImputer(strategy='most_frequent'), variables["ORDINAL_NA_NOT_ALLOWED"]),
        ("Fill_na_numeric", SimpleImputer(strategy='median'), variables["NUMERIC_NA_ALLOWED"]),

        ("Encode ordinal", OrdinalEncoder(), ordinal_variables),
        ("Encode categorical", OneHotEncoder(), categorical_variables),

        ("Remove skewness", PowerTransformer(), numeric_variables),
        ("Scale data", RobustScaler(with_centering=False), ordinal_variables + numeric_variables)],
        remainder="passthrough")

    ppc_pipeline_fitted = ppc_pipeline.fit(x_train)
    _logger.info("Successfully fitted the preprocessing pipeline")
    return ppc_pipeline_fitted


# TODO: Add additional metrics
def _hyper_param_optimization(trial, model_name:str , trial_func,
                              x_train, x_test, y_train, y_test,
                              max_overfit=0.1, cv=5):
    """Find optimal hyperparameters using bayesian hypeparameter tuning
    Function optimizes provided scoring metrics with the use of CV, taking to the account overfitting
    """

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
def tune_hyperparameters(x_train, x_test, y_train, y_test, trials_dict,
                         max_overfit=0.1, cv=5, n_trials=30, n_jobs=-1):
    """Find best model and best hyperparameters from searched parameter space"""

    studies = {}
    for model_name, trial_func in trials_dict.items():
        _logger.info(f"Finding hyperameters for {model_name}")
        sampler = TPESampler(seed=config.SEED)
        study = create_study(direction="maximize", study_name=model_name, sampler=sampler)
        study.optimize(lambda trial: _hyper_param_optimization(trial, model_name, trial_func,
                                                               x_train, x_test, y_train, y_test,
                                                               max_overfit=max_overfit, cv=cv),
                       n_trials=n_trials, n_jobs=n_jobs)

        studies[model_name] = study

    best_score, best_model, best_params = 0, "Best model doesn't exist", {}
    for study in studies.values():
        if study.best_value > best_score:
            best_model = study.study_name
            best_score = study.best_value
            best_params = study.best_params
    _logger.info(f"Best params found for: {best_model} with score: {best_score}")

    final_model = eval(best_model)(**best_params).fit(x_train, y_train)
    _logger.info("Successfully tuned hyperparameters")
    return final_model
