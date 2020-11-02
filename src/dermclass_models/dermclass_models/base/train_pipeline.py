import logging

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from dermclass_models.base.config import BaseConfig
from dermclass_models.base.pickle import Pickle
from dermclass_models.base.processing.pipeline import PpcPipeline
from dermclass_models.base.processing.preprocessors import Preprocessors


class Main:

    def __init__(self, config: BaseConfig = BaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.preprocessor = Preprocessors(self.config)
        self.ppcpipeline = PpcPipeline(self.config)

        self.pipeline = None

    def run(self):
        """Desc"""
        self.logger.info("Started training the pipeline")

        df = self.preprocessor.load_data()
        x, y = self.preprocessor.split_target(df)
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=self.config.TEST_SIZE,
                                                            random_state=self.config.SEED)

        # TODO: Check if return necessary, make these lines stick to the convention of rest of the code
        self.ppcpipeline.fit_data(x_train, x_test, y_train, y_test)
        custom_pipeline = self.ppcpipeline.fit_ppc_pipeline()

        x_train = custom_pipeline.transform(x_train)
        x_test = custom_pipeline.transform(x_test)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        ###############################

        best_model = self.ppcpipeline.tune_hyperparameters(self.config.trials_dict,
                                                           x_train, x_test, y_train, y_test,
                                                           **self.config.tuning_func_params)
        # Assemble final pipeline
        self.pipeline = Pipeline([("Preprocessing pipeline", custom_pipeline),
                                  ("Model", best_model)])

        # Save pipeline to pickle
        pickler = Pickle(self.config)
        pickler.remove_old_pipelines([])
        pickler.save_pipeline(self.pipeline)
