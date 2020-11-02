import numpy as np
import pytest
from pathlib import Path

from sklearn.pipeline import Pipeline

from dermclass_models.base.pickle import Pickle

from dermclass_models import __version__ as _version

from dermclass_models.base.train_pipeline import Main
from dermclass_models.base.config import TestingConfig
from dermclass_models.base.processing.preprocessors import Preprocessors
from dermclass_models.base.predict import Predict

from dermclass_models.structured.train_pipeline import StructuredMain
from dermclass_models.structured.config import TestingStructuredConfig
from dermclass_models.structured.processing.preprocessors import StructuredPreprocessors
from dermclass_models.structured.predict import StructuredPredict

from dermclass_models.text.train_pipeline import TextMain
from dermclass_models.text.config import TestingTextConfig
from dermclass_models.text.processing.preprocessors import TextPreprocessors
from dermclass_models.text.predict import TextPredict

# TODO: Separate these tests
# TODO: Fix logging with pytest


# @pytest.mark.dependency()
@pytest.mark.parametrize("config,main",
                         [(TestingStructuredConfig, StructuredMain), (TestingTextConfig, TextMain)])
def test_train_pipeline(config: TestingConfig, main: Main):
    """Test 'train_pipeline' function using Pytest"""

    pickler = Pickle(config)
    pickler.remove_old_pipelines([])
    main_obj = main(config)

    main_obj.run()

    pipeline_file_name = Path(f"{config.PIPELINE_TYPE}_{_version}.pkl")
    file_path = config.PICKLE_DIR / pipeline_file_name
    pipeline = pickler.load_pipeline(file_path)

    assert pipeline is not None
    assert isinstance(pipeline, Pipeline)


# @pytest.mark.dependency(depends=["test_train_pipeline"])
@pytest.mark.parametrize("config,preprocessors,predict,output_type",
                         [(TestingStructuredConfig, StructuredPreprocessors, StructuredPredict, np.int64),
                          (TestingTextConfig, TextPreprocessors, TextPredict, str)])
def test_make_prediction(config: TestingConfig, preprocessors: Preprocessors, predict: Predict, output_type):
    """Test 'make_prediction' function using Pytest"""
    pc = preprocessors(config)

    df = pc.load_data()
    x, y = pc.split_target(df)
    single_x = x[:1]

    predictor = predict(config)
    prediction = predictor.make_prediction(single_x)[0]
    assert prediction is not None
    assert isinstance(prediction, output_type)
