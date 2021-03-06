import logging
import joblib
from typing import List, Union
from pathlib import Path
import os

from sklearn.pipeline import Pipeline as SklearnPipeline
from tensorflow.keras.models import load_model, Sequential
from dermclass_models.pipeline import TransformersModelingPipeline


from dermclass_models import __version__ as _version
from dermclass_models.validation import ValidationError, validate_variables


# TODO: Refactor to make it OOP like rest of modules
class BasePersistence:

    def __init__(self, config):
        """
        Class for saving and loading pipeline objects.
        :param config: Config object for the class
        """
        validate_variables(config)

        self.config = config
        self.pipeline_version = _version
        self.logger = logging.getLogger(__name__)

    def load_pipeline(self, backend: str = None, path: Path = None)\
            -> Union[TransformersModelingPipeline, SklearnPipeline, Sequential]:
        """
        Function for loading pipeline from given path using provided backend. Can be used either with set params or
        params from the config
        :param backend: Type of backend used for loading given pipeline, has to be one of ["joblib", "tf", "tfm"]
        :param path: Path to loaded file or directory
        :return: Returns a pipeline for making predictions
        """
        if backend not in ["joblib", "tf", "tfm"]:
            raise ValidationError("Please choose proper backend from ['joblib', 'tf', 'tfm']")
        path = path or self.config.PICKLE_DIR / f"{self.config.PIPELINE_TYPE}_{self.pipeline_version}"
        validate_variables(backend, path)
        if backend == "joblib":
            pipeline = joblib.load(str(path) + '.joblib')
        elif backend == "tf":
            pipeline = load_model(path)
        elif backend == "tfm":
            pipeline = TransformersModelingPipeline.load_from_pretrained(path)
        else:
            pipeline = None

        self.logger.info(f"{path.name} loaded")
        return pipeline

    @staticmethod
    def _remove_files(path: Path):
        """
        Utility function to remove files from given path
        :param path: Path to directory to remove all files from
        """
        validate_variables(path)

        for root, dirs, files in os.walk(str(path), topdown=False):
            for file_name in files:
                (Path(root) / file_name).unlink()
            for dir_name in dirs:
                (Path(root) / dir_name).rmdir()
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()

    def remove_old_pipelines(self,
                             dir_path: Path = None,
                             pipelines_to_keep: List[Path] = None,
                             pipeline_type: str = None):
        """
        Remove old pipelines from directory using either config or list of files not to remove from pickle directory
        :param dir_path: Directory from which files old pipelines should be removed
        :param pipelines_to_keep: A list of paths to files or directories that shouldn't be removed
        :param pipeline_type: Type of pipeline to be removed as a string name from config.py
        """
        pipeline_type = pipeline_type or self.config.PIPELINE_TYPE
        pipelines_to_keep = pipelines_to_keep or []
        dir_path = dir_path or self.config.PICKLE_DIR

        do_not_delete = pipelines_to_keep + [Path(p) for p in ['__init__.py', ".gitkeep"]]

        for file in dir_path.iterdir():
            if file.name not in do_not_delete and file.name.startswith(pipeline_type):
                self._remove_files(file)
                self.logger.info(f"{file} removed")

    def save_pipeline(self, pipeline_object: Union[TransformersModelingPipeline, SklearnPipeline, Sequential],
                      backend: str = None,
                      path: Path = None):
        """
        A function for saving pipeline using provided backend to given path
        :param pipeline_object: A pipeline object to save
        :param backend: Type of backend used for saving given pipeline, has to be one of ["joblib", "tf", "tfm"]
        :param path: Path to save file or directory
        """
        if backend not in ["joblib", "tf", "tfm"]:
            raise ValidationError("Please choose proper backend from ['joblib', 'tf', 'tfm']")
        path = path or self.config.PICKLE_DIR / f"{self.config.PIPELINE_TYPE}_{self.pipeline_version}"
        validate_variables(pipeline_object, backend, path)

        self.remove_old_pipelines()

        if backend == "joblib":
            joblib.dump(pipeline_object, str(path) + ".joblib")
        if backend == "tf":
            pipeline_object.save(path)
        if backend == "tfm":
            pipeline_object.processing_pipeline.tokenizer.save_pretrained(path)
            pipeline_object.model.save_pretrained(path)

        self.logger.info(f"Saved pipeline {str(pipeline_object)}, to path {path}")
