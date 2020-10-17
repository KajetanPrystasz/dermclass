from flask import Flask
from flask_restful import Api

import logging

from dermclass_api.extensions import db, ma
from dermclass_api.resources.utils import utils_blueprint
from dermclass_api.config import DevelopmentConfig, TestingConfig
from dermclass_api.resources.structured_prediction import StructuredPrediction

_logger = logging.getLogger(__name__)


def create_app(config_object=DevelopmentConfig):
    """Create a flask app instance."""

    flask_app = Flask('api')
    flask_app.config.from_object(config_object)
    flask_app.register_blueprint(utils_blueprint)

    db.init_app(flask_app)
    ma.init_app(flask_app)

    @flask_app.before_first_request
    def create_tables():
        db.create_all()

    api = Api(flask_app)
    api.add_resource(StructuredPrediction, '/structured_prediction/<string:prediction_id>')
    _logger.debug('App instance has been created')

    return flask_app
