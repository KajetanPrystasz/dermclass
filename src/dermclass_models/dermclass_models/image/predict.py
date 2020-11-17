import logging
from pathlib import Path

import numpy as np

from dermclass_models.image.config import ImageConfig
from dermclass_models import __version__ as _version
from dermclass_models.base.pickle import Pickle

class ImagePredict:

    def __init__(self, config: ImageConfig = ImageConfig):
        self.config = config
        self.pipeline_version = _version
        self.logger = logging.getLogger(__name__)
        self.pickler = Pickle(self.config)

    def make_tf_prediction(self, input_data, img_shape):
        """Make for the input_data"""

        pipeline_file_name = Path(f"{self.config.PIPELINE_TYPE}_{_version}.h5")
        pipeline_path = self.config.PICKLE_DIR / pipeline_file_name
        pipeline = self.pickler.load_pipeline(pipeline_path, from_pck=False)

        data = np.resize(input_data, img_shape)
        data = np.expand_dims(data, 0)

        prediction = pipeline.predict(data)

        prediction_string = self.config.DISEASES[prediction.argmax()]

        self.logger.info(f"Made predictions with model version: {_version}"
                         f"Inputs: {input_data} "
                         f"Predictions: {prediction_string}")

        return prediction_string
