import pytest

import pandas as pd
import numpy as np
import tensorflow as tf


class TestingConfig:
    pass


@pytest.fixture()
def testing_config():
    return TestingConfig


@pytest.fixture()
def structured_training_set():
    columns = ["erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon", "polygonal_papules",
               "follicular_papules", "oral_mucosal_involvement", "knee_and_elbow_involvement", "scalp_involvement",
               "family_history", "melanin_incontinence", "eosinophils_in_the_infiltrate", "pnl_infiltrate",
               "fibrosis_of_the_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis",
               "clubbing_of_the_rete_ridges", "elongation_of_the_rete_ridges",
               "thinning_of_the_suprapapillary_epidermis", "spongiform_pustule", "munro_microabcess",
               "focal_hypergranulosis", "disappearance_of_the_granular_layer",
               "vacuolisation_and_damage_of_basal_layer", "spongiosis", "saw_tooth_appearance_of_retes",
               "follicular_horn_plug", "perifollicular_parakeratosis", "inflammatory_monoluclear_inflitrate",
               "band_like_infiltrate", "age", "target"]
    values = [[2, 2, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 2, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 55, 2],
              [2, 2, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 2,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 53, 1]]
    df = pd.DataFrame(np.array(values), columns=columns)
    return df


@pytest.fixture()
def train_dataset():
    dataset = tf.data.Dataset.range(10).batch(1)
    return dataset
