import logging
import joblib
from typing import List, Union
from pathlib import Path
import os

from sklearn.pipeline import Pipeline as SklearnPipeline
from tensorflow.keras.models import load_model, Sequential
from dermclass_models2.pipeline import TransformersModelingPipeline


from dermclass_models2 import __version__ as _version
from dermclass_models2.config import BaseConfig
from dermclass_models2.validation import ValidationError


# TODO: Add zipping and unzipping for pretrained models
class BasePersistence:

    def __init__(self, config: BaseConfig):
        self.config = config
        self.pipeline_version = _version
        self.logger = logging.getLogger(__name__)

    def load_pipeline(self, backend: str = None, path: Path = None)\
            -> Union[TransformersModelingPipeline, SklearnPipeline, Sequential]:
        if backend not in ["joblib", "tf", "tfm"]:
            raise ValidationError("Please choose proper backend from ['joblib', 'tf', 'tfm']")

        path = path or self.config.PICKLE_DIR / f"{self.config.PIPELINE_TYPE}_{self.pipeline_version}"

        if backend == "joblib":
            pipeline = joblib.load(path + ".joblib")

        if backend == "tf":
            pipeline = load_model(path)

        if backend == "tfm":
            pipeline = TransformersModelingPipeline.load_from_pretrained(path)

        self.logger.info(f"{path.name} loaded")
        return pipeline

    @staticmethod
    def _remove_files(path: Path):
        for root, dirs, files in os.walk(path, topdown=False):
            for file_name in files:
                (Path(root) / file_name).unlink()
            for dir_name in dirs:
                (Path(root) / dir_name).rmdir()
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()

    def remove_old_pipelines(self, pipelines_to_keep: List[Path] = None):
        pipelines_to_keep = pipelines_to_keep or []
        do_not_delete = pipelines_to_keep + [Path(p) for p in ['__init__.py', ".gitkeep"]]

        for file in self.config.PICKLE_DIR.iterdir():
            if file.name not in do_not_delete and file.name.startswith(self.config.PIPELINE_TYPE):
                self._remove_files(file)
                self.logger.info(f"{file} removed")

    def save_pipeline(self, pipeline_object: Union[TransformersModelingPipeline, SklearnPipeline, Sequential],
                      backend: str = None,
                      path: Path = None):
        if backend not in ["joblib", "tf", "tfm"]:
            raise ValidationError("Please choose proper backend from ['joblib', 'tf', 'tfm']")

        self.remove_old_pipelines()

        path = path or self.config.PICKLE_DIR / f"{self.config.PIPELINE_TYPE}_{self.pipeline_version}"

        if backend == "joblib":
            joblib.dump(pipeline_object, path.with_suffix('.joblib'))
        if backend == "tf":
            pipeline_object.save(path)
        if backend == "tfm":
            pipeline_object.processing_pipeline.tokenizer.save_pretrained(path)
            pipeline_object.model.save_pretrained(path)

        self.logger.info(f"Saved pipeline {str(pipeline_object)}, to path {path}")
