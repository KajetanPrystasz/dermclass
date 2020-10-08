import dermclass_api
import pathlib
import logging
import sys

PACKAGE_ROOT = pathlib.Path(dermclass_api.__file__).resolve().parent
log_format = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s")
LOG_DIR = PACKAGE_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'api.log'


def get_console_handler() -> logging:
    """Use stream handler for logger to log to stdout using proper format"""

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    return console_handler


def get_file_handler():
    """Use file handler for logger to log to file using proper format"""

    file_handler = logging.handlers.TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.WARNING)
    return file_handler


class BaseConfig:
    pass


class DevelopmentConfig(BaseConfig):
    pass


class Testing(BaseConfig):
    pass


class ProductionConfig(BaseConfig):
    pass
