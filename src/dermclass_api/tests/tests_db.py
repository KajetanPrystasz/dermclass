from dermclass_api import config
from dermclass_api.models.structured_prediction import StructuredPredictionModel
import json
import pytest_dependency

@pytest_dependency.depends("test_post_endpoint", scope="module")
def test_add_to_db(flask_test_client, db_connection):
    # Given
    n_rows = db_connection.query(StructuredPredictionModel).count()
    data = json.load(config.TEST_DB_FILE)

    # When
    flask_test_client.post('structured_prediction/99999', json=data)

    n_rows_updated = db_connection.query(StructuredPredictionModel).count()

    # Then
    assert n_rows < n_rows_updated


@pytest_dependency.depends(["test_get_endpoint", "test_post_endpoint"], scope="module")
def test_get_from_db(flask_test_client, db_connection):
    # Given
    n_rows = db_connection.query(StructuredPredictionModel).count()
    data = json.load(config.TEST_DB_FILE)

    # When
    flask_test_client.post('structured_prediction/99999', json=data)

    data_returned = json.load(flask_test_client.get('structured_prediction/99999'))

    # Then
    assert isinstance(data_returned, dict)



@pytest_dependency.depends(["test_remove_endpoint", "test_post_endpoint"], scope="module")
def test_delete_from_db(flask_test_client, db_connection):
    # Given
    n_rows = db_connection.query(StructuredPredictionModel).count()
    data = json.load(config.TEST_DB_FILE)

    # When
    flask_test_client.post('structured_prediction/99999', json=data)

    n_rows_updated = db_connection.query(StructuredPredictionModel).count()

    # Then
    assert n_rows < n_rows_updated
