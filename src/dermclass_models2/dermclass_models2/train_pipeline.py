import logging
import abc
import argparse

from dermclass_models2.preprocessing import StructuredPreprocessor, TextPreprocessors, ImagePreprocessors
from dermclass_models2.pipeline import StructuredPipeline, TextPipeline, ImagePipeline
from dermclass_models2.config import StructuredConfig, TextConfig, ImageConfig
from dermclass_models2.persistence import BasePersistence

from sklearn.pipeline import Pipeline


class _BaseTrainPipeline(abc.ABC):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.modeling_pipeline = None

    def _save_modeling_pipeline(self, modeling_pipeline, backend):
        modeling_pipeline = self.modeling_pipeline or modeling_pipeline
        persister = BasePersistence(self.config)
        persister.remove_old_pipelines([])
        persister.save_pipeline(modeling_pipeline, backend)


class StructuredTrainPipeline(_BaseTrainPipeline):
    def __init__(self, config: StructuredConfig = StructuredConfig):
        super().__init__(config)

        self.preprocessor = StructuredPreprocessor(self.config)
        self.pipeline = StructuredPipeline(self.config)

    def run(self):
        x_train, x_test, y_train, y_test = self.preprocessor.load_data()
        self.pipeline.fit_structured_data(x_train, x_test, y_train, y_test)

        modeling_pipeline = self.pipeline.get_modeling_pipeline()

        self._save_modeling_pipeline(modeling_pipeline, "joblib")


class ImageTrainPipeline(_BaseTrainPipeline):
    def __init__(self, config: ImageConfig = ImageConfig):
        super().__init__(config)

        self.preprocessor = ImagePreprocessors(self.config)
        self.pipeline = ImagePipeline(self.config)

    def run(self):
        train_dataset, validation_dataset, test_dataset = self.preprocessor.load_data()
        self.pipeline.fit_datasets(train_dataset, validation_dataset, test_dataset)

        self.pipeline.set_img_size_and_model_obj(self.preprocessor.img_size, self.preprocessor.model)
        modeling_pipeline = self.pipeline.get_modeling_pipeline()

        self._save_modeling_pipeline(modeling_pipeline, "tf")


class TextTrainPipeline(_BaseTrainPipeline):
    def __init__(self, config: TextConfig = TextConfig):
        super().__init__(config)

        self.preprocessor = TextPreprocessors(self.config)

        self.pipeline = TextPipeline(self.config)

    def run(self):
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
        if isinstance(modeling_pipeline, Pipeline):
            backend = "joblib"
        else:
            backend = "tfm"

        self._save_modeling_pipeline(modeling_pipeline, backend)


def get_parser():
    parser = argparse.ArgumentParser(description='Parse pipeline types to train')
    parser.add_argument('pipeline_types', metavar="N", type=str, nargs='+', help='Name of pipeline to train')
    return parser


def run_controller(pipeline_types=('structured', 'text', "image")):
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
    parser = get_parser()
    pipeline_types_args = tuple(vars(parser.parse_args())["pipeline_types"])
    run_controller(pipeline_types_args)
