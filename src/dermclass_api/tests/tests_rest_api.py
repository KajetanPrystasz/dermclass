from dermclass_structured import __version__ as model_version
from dermclass_structured import config
from dermclass_structured.processing.preprocessors import load_data
import json
import numpy as np
import pytest_dependency

def test_post_endpoint(flask_test_client):
    _, _, test_data = load_data(file_name=config.DATA_PATH)
    post_json = test_data[0].to_json(orient='records')

    # When
    response = flask_test_client.post('structured_prediction/99999',
                                      json=json.loads(post_json))

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']
    assert prediction is not None
    assert isinstance(prediction, np.int64)
    assert response_version == model_version


@pytest_dependency.depends("test_health_endpoint", scope="module")
def test_get_endpoint(flask_test_client):

    # When
    response = flask_test_client.get('structured_prediction/99999')

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']
    assert prediction is not None
    assert isinstance(prediction, np.int64)
    assert response_version == model_version

@pytest_dependency.depends("test_health_endpoint", scope="module")
def test_delete_endpoint(flask_test_client):

    # When
    response = flask_test_client.get('structured_prediction/99999')

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']
    assert prediction is not None
    assert isinstance(prediction, np.int64)
    assert response_version == model_version

