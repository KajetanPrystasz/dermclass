import json
from dermclass_models import __version__ as model_version


class TestStructuredPredictionResource:
    test_input_data = {"erythema": 2, "scaling": 2, "definite_borders": 2, "itching": 0, "koebner_phenomenon": 3,
                       "polygonal_papules": 0, "follicular_papules": 0, "oral_mucosal_involvement": 0,
                       "knee_and_elbow_involvement": 0, "scalp_involvement": 1, "family_history": 0,
                       "melanin_incontinence": 0, "eosinophils_in_the_infiltrate": 0, "pnl_infiltrate": 0,
                       "fibrosis_of_the_papillary_dermis": 3, "exocytosis": 2, "acanthosis": 0, "hyperkeratosis": 0,
                       "parakeratosis": 0, "clubbing_of_the_rete_ridges": 0, "elongation_of_the_rete_ridges": 0,
                       "thinning_of_the_suprapapillary_epidermis": 0, "spongiform_pustule": 0,
                       "munro_microabcess": 0,
                       "focal_hypergranulosis": 0, "disappearance_of_the_granular_layer": 0,
                       "vacuolisation_and_damage_of_basal_layer": 0, "spongiosis": 3,
                       "saw_tooth_appearance_of_retes": 0, "follicular_horn_plug": 0,
                       "perifollicular_parakeratosis": 0,
                       "inflammatory_monoluclear_inflitrate": 1, "band_like_infiltrate": 0, "age": 55}

    def test_get(self, flask_test_client, testing_config):

        post_json = json.dumps(TestStructuredPredictionResource.test_input_data)
        _ = flask_test_client.post('structured_prediction/99999',
                                   json=json.loads(post_json))

        response = flask_test_client.get('structured_prediction/99999')

        # Then
        assert response.status_code == 200
        response_json = json.loads(response.data)
        prediction_proba = response_json['prediction_proba']
        prediction_string = response_json['prediction_string']
        response_version = response_json['version']

        assert prediction_proba
        assert prediction_string
        assert response_version == model_version

    def test_post(self, flask_test_client, testing_config):

        post_json = json.dumps(TestStructuredPredictionResource.test_input_data)
        response = flask_test_client.post('structured_prediction/99999',
                                          json=json.loads(post_json))
        assert response.status_code == 200

    def test_delete(self, flask_test_client, testing_config):
        post_json = json.dumps(TestStructuredPredictionResource.test_input_data)
        _ = flask_test_client.post('structured_prediction/99999',
                                   json=json.loads(post_json))
        response = flask_test_client.delete('structured_prediction/99999')

        assert response.status_code == 200


class TestTextPredictionResource:
    test_input_data = {"text" : "Hello, I'm so very sick person"}

    def test_get(self, flask_test_client, testing_config):

        post_json = json.dumps(TestTextPredictionResource.test_input_data)
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

    def test_post(self, flask_test_client, testing_config):

        post_json = json.dumps(TestTextPredictionResource.test_input_data)
        response = flask_test_client.post('text_prediction/99999',
                                          json=json.loads(post_json))
        assert response.status_code == 200

    def test_delete(self, flask_test_client, testing_config):
        post_json = json.dumps(TestTextPredictionResource.test_input_data)
        _ = flask_test_client.post('text_prediction/99999',
                                   json=json.loads(post_json))
        response = flask_test_client.delete('text_prediction/99999')

        assert response.status_code == 200


# TODO: Fix testing of file predictions after providing persistent storage
class TestImagePredictionResource:
    pass
