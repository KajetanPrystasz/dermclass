import logging

from dermclass_structured.processing.pipeline import fit_ppc_pipeline, tune_hyperparameters
from dermclass_structured.processing.preprocessors import load_data
from dermclass_structured.pickles.pickle_handling import remove_old_pipelines, save_pipeline
from dermclass_structured import config

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

_logger = logging.getLogger(__name__)


def run():
    _logger.info("Started training the pipeline")

    # Load data
    x, y, df = load_data(config.DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=config.SEED)

    # Fit preprocessing pipeline and transform data
    ppc_pipeline = fit_ppc_pipeline(x_train)

    # Transform data
    x_train = ppc_pipeline.transform(x_train)
    x_test = ppc_pipeline.transform(x_test)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Find optimal hyperparameters /// Increase number of trials for better results
    best_model = tune_hyperparameters(x_train, x_test, y_train, y_test, config.trials_dict,
                                      n_trials=20, n_jobs=1, max_overfit=0.05, cv=3)

    # Assemble final pipeline
    final_pipeline = Pipeline([("Ppc_pipeline", ppc_pipeline),
                               ("Model", best_model)])

    # Save pipeline to pickle
    remove_old_pipelines([])
    save_pipeline(final_pipeline)
