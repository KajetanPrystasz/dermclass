from flask import Flask
from flask_restful import Api

import logging

from dermclass_api.extensions import db, ma
from dermclass_api.utils_resources import utils_blueprint
from dermclass_api.config import DevelopmentConfig
from dermclass_api.prediction_resources import (StructuredPredictionResource,
                                                TextPredictionResource,
                                                ImagePredictionResource)

_logger = logging.getLogger(__name__)


def create_app(config_object=DevelopmentConfig):
    """Create a flask app instance."""

    flask_app = Flask('app')
    flask_app.config.from_object(config_object)
    flask_app.register_blueprint(utils_blueprint)

    db.init_app(flask_app)
    ma.init_app(flask_app)

    @flask_app.before_first_request
    def create_tables():
        db.create_all()

    api = Api(flask_app)

    api.add_resource(StructuredPredictionResource, '/structured_prediction/<string:prediction_id>')
    api.add_resource(TextPredictionResource, '/text_prediction/<string:prediction_id>')
    api.add_resource(ImagePredictionResource, '/image_prediction/<string:prediction_id>')

    _logger.debug('App instance has been created')

    return flask_app
