from flask import jsonify, request, Blueprint
from dermclass_structured import __version__ as model_version
from dermclass_api import logger as _logger, __version__ as api_version

utils_blueprint = Blueprint("utils_blueprint", __name__)


@utils_blueprint.route('/', methods=['GET'])
@utils_blueprint.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'health status OK', 200


@utils_blueprint.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': model_version,
                        'api_version': api_version})
