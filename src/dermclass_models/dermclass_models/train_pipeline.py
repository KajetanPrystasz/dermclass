import logging
import abc
import argparse
from typing import Union

import tensorflow as tf

from dermclass_models.preprocessing import StructuredPreprocessor, TextPreprocessors, ImagePreprocessors
from dermclass_models.pipeline import StructuredPipeline, TextPipeline, ImagePipeline
from dermclass_models.config import StructuredConfig, TextConfig, ImageConfig
from dermclass_models.persistence import BasePersistence
from dermclass_models.pipeline import TransformersModelingPipeline
from dermclass_models.validation import validate_variables

from sklearn.pipeline import Pipeline as SklearnPipeline

Sequential = tf.keras.models.Sequential


class _BaseTrainPipeline(abc.ABC):
    def __init__(self, config):
        """
        Abstract base class for training pipeline and saving it
        :param config: Config object for the class
        """
        validate_variables(config)

        self.config = config
        self.logger = logging.getLogger(__name__)
        self.modeling_pipeline = None

    def _save_modeling_pipeline(self,
                                modeling_pipeline: Union[TransformersModelingPipeline, SklearnPipeline, Sequential],
                                backend: str):
        """
        Utility function to save modeling pipeline using provided backend
        :param modeling_pipeline: A modeling pipeline to save
        :param backend: Type of backend used for loading given pipeline, has to be one of ["joblib", "tf", "tfm"]
        """
        modeling_pipeline = self.modeling_pipeline or modeling_pipeline
        validate_variables(modeling_pipeline, backend)
        if not self.config:
            raise RuntimeError("No config object fitted")

        persister = BasePersistence(self.config)
        persister.remove_old_pipelines([])
        persister.save_pipeline(modeling_pipeline, backend)


class StructuredTrainPipeline(_BaseTrainPipeline):

    def __init__(self, config: StructuredConfig = StructuredConfig):
        """
        A class used for training structured pipeline and saving it
        :param config: Config object for the class
        """
        super().__init__(config)
        if not self.config:
            raise RuntimeError("No config object fitted")

        self.preprocessor = StructuredPreprocessor(self.config)
        self.pipeline = StructuredPipeline(self.config)

    def run(self):
        """Function to run training of the structured pipeline"""
        if not self.preprocessor:
            raise RuntimeError("No preprocessor object fitted")
        if not self.pipeline:
            raise RuntimeError("No pipeline object fitted")

        x_train, x_test, y_train, y_test = self.preprocessor.load_data()
        self.pipeline.fit_structured_data(x_train, x_test, y_train, y_test)

        modeling_pipeline = self.pipeline.get_modeling_pipeline()

        self._save_modeling_pipeline(modeling_pipeline, "joblib")


class ImageTrainPipeline(_BaseTrainPipeline):
    def __init__(self, config: ImageConfig = ImageConfig):
        """
        A class used for training image pipeline and saving it
        :param config: Config object for the class
        """
        super().__init__(config)
        if not self.config:
            raise RuntimeError("No config object fitted")

        self.preprocessor = ImagePreprocessors(self.config)
        self.pipeline = ImagePipeline(self.config)

    def run(self):
        """Function to run training of the structured pipeline"""
        if not self.preprocessor:
            raise RuntimeError("No preprocessor object fitted")
        if not self.pipeline:
            raise RuntimeError("No pipeline object fitted")

        train_dataset, validation_dataset, test_dataset = self.preprocessor.load_data()
        self.pipeline.fit_datasets(train_dataset, validation_dataset, test_dataset)
        self.pipeline.set_img_size_and_model_obj(self.preprocessor.img_size, self.preprocessor.model)
        modeling_pipeline = self.pipeline.get_modeling_pipeline()

        self._save_modeling_pipeline(modeling_pipeline, "tf")


class TextTrainPipeline(_BaseTrainPipeline):
    def __init__(self, config: TextConfig = TextConfig):
        """
        A class used for training text pipeline and saving it
        :param config: Config object for the class
        """
        super().__init__(config)
        if not self.config:
            raise RuntimeError("No config object fitted")

        self.preprocessor = TextPreprocessors(self.config)
        self.pipeline = TextPipeline(self.config)

    def run(self):
        """Function to train text pipelines, choose if sklearn or transformer is better one and save it"""
        if not self.preprocessor:
            raise RuntimeError("No preprocessor object fitted")
        if not self.pipeline:
            raise RuntimeError("No pipeline object fitted")

        x_train, x_test, y_train, y_test = self.preprocessor.load_data(get_datasets=False)
        self.pipeline.fit_structured_data(x_train, x_test, y_train, y_test)
        sklearn_modeling_pipeline = self.pipeline.get_modeling_pipeline(use_sklearn=True)

        train_dataset, validation_dataset, test_dataset = self.preprocessor.load_data(get_datasets=True)
        self.pipeline.fit_datasets(train_dataset, validation_dataset, test_dataset)
        transformers_modeling_pipeline = self.pipeline.get_modeling_pipeline(use_sklearn=False)

        modeling_pipeline = self.pipeline.get_best_modeling_pipeline_type(transformers_modeling_pipeline,
                                                                          sklearn_modeling_pipeline,
                                                                          x_test,
                                                                          y_test,
                                                                          test_dataset)
        if isinstance(modeling_pipeline, SklearnPipeline):
            backend = "joblib"
        else:
            backend = "tfm"

        self._save_modeling_pipeline(modeling_pipeline, backend)


def _get_parser() -> argparse.ArgumentParser:
    """Utility function to get parser to run script"""
    parser = argparse.ArgumentParser(description='Parse pipeline types to train')
    parser.add_argument('--pipeline_types', metavar="N", type=str, nargs='+', help='Name of pipeline to train')
    return parser


def run_controller(pipeline_types=('structured', 'text', "image")):
    """
    Controller to run training of given types of pipelines
    :param pipeline_types: Type of pipeline to train
    """
    if not any(i in pipeline_types for i in ['structured', 'text', "image"]):
        raise RuntimeError("No pipeline types to train inputted")

    for pipeline_type in pipeline_types:
        if pipeline_type == "structured":
            sm = StructuredTrainPipeline()
            sm.run()
        if pipeline_type == "text":
            tm = TextTrainPipeline()
            tm.run()

        if pipeline_type == "image":
            im = ImageTrainPipeline()
            im.run()


if __name__ == "__main__":
    parser = _get_parser()
    pipeline_types_args = tuple(vars(parser.parse_args())["pipeline_types"])
    run_controller(pipeline_types_args)
