from dermclass_structured.train_pipeline import run
from dermclass_structured.pickles.pickle_handling import load_pipeline, remove_old_pipelines
from dermclass_structured import config, __version__ as _version
from dermclass_structured.predict import make_prediction
from dermclass_structured.processing.preprocessors import load_data

import numpy as np
from sklearn.pipeline import Pipeline
import pytest


# TODO: Fix logging with pytest
@pytest.mark.dependency()
def test_train_pipeline():
    """Test 'train_pipeline' function using Pytest"""

    remove_old_pipelines([])

    run(testing=True)
    pipeline_file_name = f"{config.PIPELINE_NAME}_{_version}.pkl"
    pipeline = load_pipeline(file_name=pipeline_file_name)

    assert pipeline is not None
    assert isinstance(pipeline, Pipeline)


@pytest.mark.dependency(depends=["test_train_pipeline"])
def test_make_prediction():
    """Test 'make_prediction' function using Pytest"""

    x, y, df = load_data(config.DATA_PATH)
    single_x = x[:1]

    prediction = make_prediction(single_x)[0]
    assert prediction is not None
    assert isinstance(prediction, np.int64)
