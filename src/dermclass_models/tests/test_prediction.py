import pytest

import numpy as np
from PIL import Image

from dermclass_models.prediction import StructuredPrediction, ImagePrediction, TextPrediction


@pytest.mark.integration
class TestStructuredPrediction:
    def test_make_prediction(self, testing_config):
        testing_config.PIPELINE_TYPE = "structured_pipeline"
        testing_config.VARIABLE_ORDER = [
            'erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 'polygonal_papules',
            'follicular_papules', 'oral_mucosal_involvement', 'knee_and_elbow_involvement', 'scalp_involvement',
            'family_history', 'melanin_incontinence', 'eosinophils_in_the_infiltrate', 'pnl_infiltrate',
            'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis',
            'clubbing_of_the_rete_ridges', 'elongation_of_the_rete_ridges', 'thinning_of_the_suprapapillary_epidermis',
            'spongiform_pustule', 'munro_microabcess', 'focal_hypergranulosis', 'disappearance_of_the_granular_layer',
            'vacuolisation_and_damage_of_basal_layer', 'spongiosis', 'saw_tooth_appearance_of_retes',
            'follicular_horn_plug', 'perifollicular_parakeratosis', 'inflammatory_monoluclear_inflitrate',
            'band_like_infiltrate', "age", "target"]

        testing_config.NA_VALIDATION_VAR_DICT = {
            "NUMERIC_NA_ALLOWED": ["age"],
            "NUMERIC_NA_NOT_ALLOWED": [],

            "CATEGORICAL_NA_ALLOWED": [],
            "CATEGORICAL_NA_NOT_ALLOWED": [],

            "ORDINAL_NA_ALLOWED": [],
            "ORDINAL_NA_NOT_ALLOWED": [var for var in testing_config.VARIABLE_ORDER if var not in ["age", "target"]]
        }

        predictor = StructuredPrediction(testing_config)
        test_input_data = {"erythema": 2, "scaling": 2, "definite_borders": 2, "itching": 0, "koebner_phenomenon": 3,
                           "polygonal_papules": 0, "follicular_papules": 0, "oral_mucosal_involvement": 0,
                           "knee_and_elbow_involvement": 0, "scalp_involvement": 1, "family_history": 0,
                           "melanin_incontinence": 0, "eosinophils_in_the_infiltrate": 0, "pnl_infiltrate": 0,
                            "fibrosis_of_the_papillary_dermis": 3, "exocytosis": 2, "acanthosis": 0, "hyperkeratosis": 0,
                           "parakeratosis": 0, "clubbing_of_the_rete_ridges": 0, "elongation_of_the_rete_ridges": 0,
                           "thinning_of_the_suprapapillary_epidermis": 0, "spongiform_pustule": 0, "munro_microabcess": 0,
                           "focal_hypergranulosis": 0, "disappearance_of_the_granular_layer": 0,
                           "vacuolisation_and_damage_of_basal_layer": 0, "spongiosis": 3,
                           "saw_tooth_appearance_of_retes": 0, "follicular_horn_plug": 0, "perifollicular_parakeratosis": 0,
                           "inflammatory_monoluclear_inflitrate": 1, "band_like_infiltrate": 0, "age": 55}
        prediction_probabilities, prediction_string = predictor.make_prediction(test_input_data)
        assert prediction_probabilities
        assert prediction_string


@pytest.mark.integration
class TestImagePrediction:
    def test_make_prediction(self, testing_config):
        testing_config.PIPELINE_TYPE = "image_pipeline"
        predictor = ImagePrediction(testing_config)

        image = Image.open(testing_config.PACKAGE_ROOT
                           / ".." / "tests"
                           / "test_image_dir" / "test_image_dir_subclass"
                           / "0001.jpg")
        np_image = np.array(image)
        test_input_data = {"img_array": np_image}
        prediction_probabilities, prediction_string = predictor.make_prediction(test_input_data)
        assert prediction_probabilities
        assert prediction_string


@pytest.mark.integration
class TestTextPrediction:
    def test_make_prediction(self, testing_config):
        testing_config.VARIABLE_ORDER = ["text"]
        testing_config.PIPELINE_TYPE = "text_pipeline"
        predictor = TextPrediction(testing_config)
        test_input_data = {"text": "Hello, I'm so very sick person"}
        prediction_probabilities, prediction_string = predictor.make_prediction(test_input_data)
        assert prediction_probabilities
        assert prediction_string
