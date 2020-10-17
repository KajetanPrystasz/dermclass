from dermclass_structured import __version__ as model_version
from dermclass_api import __version__ as api_version
import json
import pytest


@pytest.mark.dependency(scope="module")
def test_health_endpoint(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


def test_version_endpoint(flask_test_client):
    # When
    response = flask_test_client.get('/version')

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['model_version'] == model_version
    assert response_json['api_version'] == api_version
