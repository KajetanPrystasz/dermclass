import logging
import joblib
from typing import List

from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model

from dermclass_models import __version__ as _version
from dermclass_models.base.config import BaseConfig


class Pickle:

    def __init__(self, config: BaseConfig):
        self.config = config

        self.pipeline_version = _version

        self.logger = logging.getLogger(__name__)

    # TODO: Add proper type output
    def load_pipeline(self, path: str = None, from_pck=True):
        """Load pipeline from to pickle folder using file name"""

        if from_pck:
            path = path or self.config.PICKLE_DIR / f"{self.config.PIPELINE_TYPE}_{self.pipeline_version}.pkl"
            trained_model = joblib.load(filename=path)
        else:
            path = path or self.config.PICKLE_DIR / f"{self.config.PIPELINE_TYPE}_{self.pipeline_version}.h5"
            trained_model = load_model(path)

        file_name = path.name

        self.logger.info(f"{file_name} loaded")
        return trained_model

    # TODO: Add note about turning off automatic use of this function for archiving purposes
    # TODO: Make it always use same input (full path or .name?)
    def save_pipeline(self, pipeline_object: Pipeline, to_pck=True):
        """Save pipeline to pickle folder"""

        if to_pck:
            save_file_name = f"{self.config.PIPELINE_TYPE}_{self.pipeline_version}.pkl"
        else:
            save_file_name = f"{self.config.PIPELINE_TYPE}_{self.pipeline_version}.h5"
        save_path = self.config.PICKLE_DIR / save_file_name

        self.remove_old_pipelines(pipelines_to_keep=[save_file_name])

        if to_pck:
            joblib.dump(pipeline_object, save_path)
        else:
            pipeline_object.save(save_path)
        self.logger.info(f"Saved pipeline: {save_file_name}, to path: {save_path}")

    def remove_old_pipelines(self, pipelines_to_keep: List[str]):
        """Remove pickles from pickle folder except the files in files_to_keep variable"""

        do_not_delete = pipelines_to_keep + ['__init__.py', ".gitkeep"]

        for file in self.config.PICKLE_DIR.iterdir():
            if file.name not in do_not_delete and file.name.startswith(self.config.PIPELINE_TYPE):
                file.unlink()
                self.logger.info(f"{file} removed")
