import json
from dermclass_models import __version__ as model_version


class TestStructuredPredictionResource:

    def test_get(self, flask_test_client, testing_config, structured_input_data):
        post_json = json.dumps(structured_input_data)
        _ = flask_test_client.post('structured_prediction/99999',
                                   json=json.loads(post_json))

        response = flask_test_client.get('structured_prediction/99999')

        assert response.status_code == 200
        response_json = json.loads(response.data)
        prediction_proba = response_json['prediction_proba']
        prediction_string = response_json['prediction_string']
        response_version = response_json['version']

        assert prediction_proba
        assert prediction_string
        assert response_version == model_version

    def test_post(self, flask_test_client, testing_config, structured_input_data):

        post_json = json.dumps(structured_input_data)
        response = flask_test_client.post('structured_prediction/99999',
                                          json=json.loads(post_json))
        assert response.status_code == 200

    def test_delete(self, flask_test_client, testing_config, structured_input_data):
        post_json = json.dumps(structured_input_data)
        _ = flask_test_client.post('structured_prediction/99999',
                                   json=json.loads(post_json))
        response = flask_test_client.delete('structured_prediction/99999')

        assert response.status_code == 200


class TestTextPredictionResource:

    def test_get(self, flask_test_client, testing_config, text_input_data):

        post_json = json.dumps(text_input_data)
        _ = flask_test_client.post('text_prediction/99999',
                                   json=json.loads(post_json))

        response = flask_test_client.get('text_prediction/99999')

        # Then
        assert response.status_code == 200
        response_json = json.loads(response.data)
        prediction_proba = response_json['prediction_proba']
        prediction_string = response_json['prediction_string']
        response_version = response_json['version']

        assert prediction_proba
        assert prediction_string
        assert response_version == model_version

    def test_post(self, flask_test_client, testing_config, text_input_data):

        post_json = json.dumps(text_input_data)
        response = flask_test_client.post('text_prediction/99999',
                                          json=json.loads(post_json))
        assert response.status_code == 200

    def test_delete(self, flask_test_client, testing_config, text_input_data):
        post_json = json.dumps(text_input_data)
        _ = flask_test_client.post('text_prediction/99999',
                                   json=json.loads(post_json))
        response = flask_test_client.delete('text_prediction/99999')

        assert response.status_code == 200


# TODO: Fix testing of file predictions after providing persistent storage
class TestImagePredictionResource:
    def test_get(self, flask_test_client, testing_config):
        assert True

    def test_post(self, flask_test_client, testing_config):
        assert True

    def test_delete(self):
        assert True
