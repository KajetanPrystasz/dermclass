import logging
import joblib
from typing import List, Union
from pathlib import Path

from sklearn.pipeline import Pipeline as sklearn_pipeline
from tensorflow.keras.models import load_model, Sequential
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, pipeline as tfm_pipeline

from dermclass_models2 import __version__ as _version
from dermclass_models2.config import BaseConfig
from dermclass_models2.validation import ValidationError


class BasePersistence:

    def __init__(self, config: BaseConfig):
        self.config = config
        self.pipeline_version = _version
        self.logger = logging.getLogger(__name__)

    def load_pipeline(self, backend: str = None, path: Path = None)\
            -> Union[tfm_pipeline, sklearn_pipeline, Sequential]:
        if backend not in ["joblib", "tf", "tfm"]:
            raise ValidationError("Please choose proper backend from ['joblib', 'tf', 'tfm']")

        path = path or self.config.PICKLE_DIR / f"{self.config.PIPELINE_TYPE}_{self.pipeline_version}"

        if backend == "joblib":
            pipeline = joblib.load(path + ".joblib")

        if backend == "tf":
            pipeline = load_model(path)

        if backend == "tfm":
            model = TFAutoModelForSequenceClassification.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path)
            pipeline = tfm_pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, framework="tf")

        self.logger.info(f"{path.name} loaded")
        return pipeline

    def remove_old_pipelines(self, pipelines_to_keep: List[Path] = ""):
        do_not_delete = pipelines_to_keep + [Path(p) for p in ['__init__.py', ".gitkeep"]]

        for file in self.config.PICKLE_DIR.iterdir():
            if file.name not in do_not_delete and file.name.startswith(self.config.PIPELINE_TYPE):
                file.unlink()
                self.logger.info(f"{file} removed")

    def save_pipeline(self, pipeline_object: Union[tfm_pipeline, sklearn_pipeline, Sequential],
                      backend: str = None,
                      path: Path = None):
        if backend not in ["joblib", "tf", "tfm"]:
            raise ValidationError("Please choose proper backend from ['joblib', 'tf', 'tfm']")

        self.remove_old_pipelines()

        path = path or self.config.PICKLE_DIR / f"{self.config.PIPELINE_TYPE}_{self.pipeline_version}"

        if backend == "joblib":
            joblib.dump(pipeline_object, path + ".joblib")
        if backend == "tf":
            pipeline_object.save(path)
        if backend == "tfm":
            pipeline_object.save_pretrained(path)

        self.logger.info(f"Saved pipeline {str(pipeline_object)}, to path {path}")
