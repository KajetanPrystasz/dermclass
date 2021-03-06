import logging
from dermclass_models.config import BaseConfig

VERSION_PATH = BaseConfig.PACKAGE_ROOT / "VERSION"
with open(VERSION_PATH, 'r') as version_file:
    __version__ = version_file.read().strip()

# Configure logger for use in package
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(config.get_console_handler())
logger.propagate = False
