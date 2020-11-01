import logging
import joblib
from typing import List

from sklearn.pipeline import Pipeline

from dermclass_models import __version__ as _version
from dermclass_models.base.config import BaseConfig


class Pickle:

    def __init__(self, config: BaseConfig):
        self.config = config

        self.pipeline_type = self.config.pipeline_type
        self.pipeline_version = _version

        self.logger = logging.getLogger(__name__)

    # TODO: Add proper type output
    def load_pipeline(self, path: str = None):
        """Load pipeline from to pickle folder using file name"""
        if path is None:
            path = self.config.PICKLE_DIR / f"{self.pipeline_type}_{self.pipeline_version}.pkl"

        file_name = path.name
        trained_model = joblib.load(filename=path)

        self.logger.info(f"{file_name} loaded")
        return trained_model

    # TODO: Add note about turning off automatic use of this function for archiving purposes
    def save_pipeline(self, pipeline_object: Pipeline):
        """Save pipeline to pickle folder"""

        save_file_name = f"{self.pipeline_type}_{self.pipeline_version}.pkl"
        save_path = self.config.PICKLE_DIR / save_file_name

        self.remove_old_pipelines(pipelines_to_keep=[save_file_name])

        joblib.dump(pipeline_object, save_path)
        self.logger.info(f"Saved pipeline: {save_file_name}, to path: {save_path}")

    def remove_old_pipelines(self, pipelines_to_keep: List[str]):
        """Remove pickles from pickle folder except the files in files_to_keep variable"""

        do_not_delete = pipelines_to_keep + ['__init__.py']

        for file in self.config.PICKLE_DIR.iterdir():
            if file.name not in do_not_delete and file.name.startswith(self.PIPELINE_TYPE):
                file.unlink()
                self.logger.info(f"{file} removed")
