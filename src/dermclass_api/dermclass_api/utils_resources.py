from typing import Tuple
import logging

from flask import jsonify, request, Blueprint

from dermclass_models import __version__ as model_version
from dermclass_api import __version__ as api_version

utils_blueprint = Blueprint("utils_blueprint", __name__)

logger = logging.getLogger(__name__)


@utils_blueprint.route('/', methods=['GET'])
@utils_blueprint.route('/health', methods=['GET'])
def health() -> Tuple[str, int]:
    """
    Basic health check endpoint for API
    return: A tuple with information about health status and HTTP successful code
    """
    if request.method == 'GET':
        logger.info('health status OK')
        return 'health status OK', 200


@utils_blueprint.route('/version', methods=['GET'])
def version() -> dict:
    """
    Basic version endpoint for API
    return: A dict containing used model version and API version
    """
    if request.method == 'GET':
        version_dict = {'model_version': model_version,
                        'api_version': api_version}
        logger.info(version_dict)
        return jsonify(version_dict)
