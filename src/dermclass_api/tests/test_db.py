from dermclass_api import config
from dermclass_api.models.structured_prediction import StructuredPredictionModel
from dermclass_api.extensions import db
import json


def test_add_to_db(flask_test_client):

    with open(config.TEST_DB_FILE) as json_file:
        data = json.load(json_file)

    # Setup db with first request using health check
    flask_test_client.post('/health')
    n_rows = db.session.query(StructuredPredictionModel).count()

    # When
    flask_test_client.post('structured_prediction/99999', json=data)

    n_rows_updated = db.session.query(StructuredPredictionModel).count()

    # Then
    assert n_rows < n_rows_updated


def test_get_from_db(flask_test_client):
    # Given
    with open(config.TEST_DB_FILE) as json_file:
        data = json.load(json_file)

    # When
    flask_test_client.post('structured_prediction/99999', json=data)

    data_returned = json.loads(flask_test_client.get('structured_prediction/99999').data)

    # Then
    assert isinstance(data_returned, dict)


def test_delete_from_db(flask_test_client):
    # Given
    with open(config.TEST_DB_FILE) as json_file:
        data = json.load(json_file)

    # Setup db with first request using health check
    flask_test_client.post('/health')
    n_rows = db.session.query(StructuredPredictionModel).count()

    # When
    flask_test_client.post('structured_prediction/99999', json=data)

    n_rows_updated = db.session.query(StructuredPredictionModel).count()

    # Then
    assert n_rows < n_rows_updated
