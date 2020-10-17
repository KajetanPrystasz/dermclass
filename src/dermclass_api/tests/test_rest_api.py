from dermclass_structured import __version__ as model_version
from dermclass_structured import config
from dermclass_structured.processing.preprocessors import load_data
import json


def test_post_endpoint(flask_test_client):

    # When
    _, _, test_data = load_data(path=config.DATA_PATH)
    post_json = test_data.iloc[0].to_json(orient='index')
    response = flask_test_client.post('structured_prediction/99999',
                                      json=json.loads(post_json))

    # Then
    assert response.status_code == 201
    response_json = json.loads(response.data)
    prediction = response_json['target']
    response_version = response_json['version']
    assert prediction is not None
    assert isinstance(prediction, int)
    assert response_version == model_version


def test_get_endpoint(flask_test_client):

    # When
    _, _, test_data = load_data(path=config.DATA_PATH)
    post_json = test_data.iloc[0].to_json(orient='index')
    _ = flask_test_client.post('structured_prediction/99999',
                               json=json.loads(post_json))

    response = flask_test_client.get('structured_prediction/99999')

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['target']
    response_version = response_json['version']
    assert prediction is not None
    assert isinstance(prediction, int)
    assert response_version == model_version


def test_delete_endpoint(flask_test_client):

    # When
    _, _, test_data = load_data(path=config.DATA_PATH)
    post_json = test_data.iloc[0].to_json(orient='index')
    _ = flask_test_client.post('structured_prediction/99999',
                               json=json.loads(post_json))
    response = flask_test_client.get('structured_prediction/99999')

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['target']
    response_version = response_json['version']
    assert prediction is not None
    assert isinstance(prediction, int)
    assert response_version == model_version

