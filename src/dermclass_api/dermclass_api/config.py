import dermclass_api
import pathlib
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import os

PACKAGE_ROOT = pathlib.Path(dermclass_api.__file__).resolve().parent
log_format = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s")
LOG_DIR = PACKAGE_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'api.log'
TEST_DB_FILE = PACKAGE_ROOT.parent / "tests" / "test_db.json"


LABEL_MAPPING = {
    1: "psoriasis",
    2:  "seboreic dermatitis",
    3: "lichen planus",
    4: "pityriasis rosea",
    5: "cronic dermatitis",
    6: "pityriasis rubra pilaris"
}


def get_console_handler() -> logging:
    """Use stream handler for logger to log to stdout using proper format"""

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    return console_handler


def get_file_handler():
    """Use file handler for logger to log to file using proper format"""

    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.WARNING)
    return file_handler


class BaseConfig:
    DEBUG = False
    TESTING = False
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SERVER_PORT = 5000

class DevelopmentConfig(BaseConfig):
    DEVELOPMENT = True
    DEBUG = True

class TestingConfig(BaseConfig):
    TESTING = True


class ProductionConfig(BaseConfig):
    DEBUG = False
    SERVER_PORT = os.environ.get('PORT', 5000)