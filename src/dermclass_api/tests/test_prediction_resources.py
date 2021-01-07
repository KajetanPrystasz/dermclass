import json
import pytest
from dermclass_models import __version__ as model_version

structured_input_data = {"erythema": 2, "scaling": 2, "definite_borders": 2, "itching": 0, "koebner_phenomenon": 3,
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
text_input_data = {"text": "Hello, I'm so very sick person"}

structured_json = json.dumps(structured_input_data)
text_json = json.dumps(text_input_data)


class TestPredictionResource:

    @pytest.mark.parametrize("prediction_data,prediction_endpoint",
                             [((structured_json,), "structured_prediction"),
                              ((text_json,), "text_prediction")])
    def test_get(self, flask_test_client, testing_config, prediction_data, prediction_endpoint):
        _ = flask_test_client.post(f'{prediction_endpoint}/99999',
                                   json=json.loads(prediction_data[0]))

        response = flask_test_client.get(f'{prediction_endpoint}/99999')

        assert response.status_code == 200
        response_json = json.loads(response.data)
        prediction_proba = response_json['prediction_proba']
        prediction_string = response_json['prediction_string']
        response_version = response_json['version']

        assert prediction_proba
        assert prediction_string
        assert response_version == model_version

    @pytest.mark.parametrize("prediction_data,prediction_endpoint",
                             [((structured_json,), "structured_prediction"),
                              ((text_json,), "text_prediction")])
    def test_post(self, flask_test_client, testing_config, prediction_data, prediction_endpoint):
        response = flask_test_client.post(f'{prediction_endpoint}/99999',
                                          json=json.loads(prediction_data[0]))
        assert response.status_code == 200

    @pytest.mark.parametrize("prediction_data,prediction_endpoint",
                             [((structured_json,), "structured_prediction"),
                              ((text_json,), "text_prediction")])
    def test_delete(self, flask_test_client, testing_config, prediction_data, prediction_endpoint):
        _ = flask_test_client.post(f'{prediction_endpoint}/99999',
                                   json=json.loads(prediction_data[0]))
        response = flask_test_client.delete(f'{prediction_endpoint}/99999')

        assert response.status_code == 200


# TODO: Fix testing of file predictions after providing persistent storage
class TestImagePredictionResource:
    def test_get(self, flask_test_client, testing_config):
        assert True

    def test_post(self, flask_test_client, testing_config):
        assert True

    def test_delete(self):
        assert True
