from dermclass_api.extensions import db
from dermclass_api.prediction_models import StructuredPredictionModel, TextPredictionModel, ImagePredictionModel
import json
import pytest

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


class TestPredictionModel:
    def test_json(self):
        # TODO: Add this
        pass

    @pytest.mark.parametrize("prediction_data,prediction_endpoint",
                             [((structured_json,), "structured_prediction"),
                              ((text_json,), "text_prediction")])
    def test_find_by_prediction_id(self, flask_test_client, prediction_data, prediction_endpoint):
        flask_test_client.post(f'{prediction_endpoint}/99999', json=json.loads(prediction_data[0]))

        data_returned = json.loads(flask_test_client.get(f'{prediction_endpoint}/99999').data)

        assert isinstance(data_returned, dict)

    @pytest.mark.parametrize("prediction_data,prediction_endpoint,prediction_model",
                             [((structured_json,), "structured_prediction", StructuredPredictionModel),
                              ((text_json,), "text_prediction", TextPredictionModel)])
    def test_save_to_db(self, flask_test_client, prediction_data, prediction_endpoint, prediction_model):
        # Setup db with first request using health check
        flask_test_client.post('/health')
        n_rows = db.session.query(prediction_model).count()

        flask_test_client.post(f'{prediction_endpoint}/99999', json=json.loads(prediction_data[0]))
        n_rows_updated = db.session.query(prediction_model).count()

        assert n_rows < n_rows_updated

    @pytest.mark.parametrize("prediction_data,prediction_endpoint,prediction_model",
                             [((structured_json,), "structured_prediction", StructuredPredictionModel),
                              ((text_json,), "text_prediction", TextPredictionModel)])
    def test_delete_from_db(self, flask_test_client, prediction_data, prediction_endpoint, prediction_model):
        # Setup db with first request using health check
        flask_test_client.post('/health')
        n_rows = db.session.query(prediction_model).count()

        flask_test_client.post(f'{prediction_endpoint}/99999', json=json.loads(prediction_data[0]))
        n_rows_updated = db.session.query(prediction_model).count()

        assert n_rows < n_rows_updated
