import pathlib
import logging
import sys
import os
import abc
from logging.handlers import TimedRotatingFileHandler

import dermclass_api

PACKAGE_ROOT = pathlib.Path(dermclass_api.__file__).resolve().parent
log_format = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s")
LOG_DIR = PACKAGE_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'api.log'


LABEL_MAPPING = {
    1: "psoriasis",
    2:  "seboreic dermatitis",
    3: "lichen planus",
    4: "pityriasis rosea",
    5: "cronic dermatitis",
    6: "pityriasis rubra pilaris"
}


def get_console_handler() -> logging:
    """
    Use stream handler for logger to log to stdout using proper format
    return: Logger object with console handling
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    return console_handler


def get_file_handler() -> logging:
    """
    Use file handler for logger to log to file using proper format
    return: Logger object with file handling
    """
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.WARNING)
    return file_handler


class BaseConfig(abc.ABC):
    DEBUG = False
    TESTING = False
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SERVER_PORT = 5000
    SECRET_KEY = "secret_key"
    SESSION_TYPE = "filesystem"


class DevelopmentConfig(BaseConfig):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(BaseConfig):
    TESTING = True
    TEST_FILE_DIR = PACKAGE_ROOT / ".." / "tests"


class ProductionConfig(BaseConfig):
    DEBUG = False
    SERVER_PORT = os.environ.get('PORT', 5000)
