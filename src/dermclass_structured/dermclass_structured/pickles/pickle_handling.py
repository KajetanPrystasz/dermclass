from dermclass_structured import config, __version__ as _version
import logging
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from typing import List

_logger = logging.getLogger(__name__)


def save_pipeline(pipeline_object: Pipeline):
    """Save pipeline to pickle folder"""

    save_file_name = f"{config.PIPELINE_NAME}_{_version}.pkl"
    save_path = config.PICKLE_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])

    joblib.dump(pipeline_object, save_path)
    _logger.info(f"Saved pipeline: {save_file_name}, to path: {save_path}")


def load_pipeline(file_name: str) -> XGBClassifier:
    """Load pipeline from to pickle folder using file name"""

    path = config.PICKLE_DIR / file_name
    trained_model = joblib.load(filename=path)
    _logger.info(f"{file_name} loaded")
    return trained_model


def remove_old_pipelines(files_to_keep: List[str]):
    """Remove pickles from pickle folder except the files in files_to_keep variable"""

    _logger.info("Start removing old pipelines")
    do_not_delete = files_to_keep + ['__init__.py', "pickle_handling.py", "__pycache__"]
    for file in config.PICKLE_DIR.iterdir():
        if file.name not in do_not_delete:
            file.unlink()
            _logger.info(f"{file} removed")
