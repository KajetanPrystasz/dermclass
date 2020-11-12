import logging

import tensorflow as tf

from dermclass_models.image.config import ImageConfig
from dermclass_models.image.processing.pipeline import ImagePipeline
from dermclass_models.image.processing.preprocessors import ImagePreprocessors
from dermclass_models.base.pickle import Pickle


class ImageMain:

    def __init__(self, config: ImageConfig = ImageConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info("Started training the pipeline")

        preprocessors = ImagePreprocessors(self.config)
        train_dataset, validation_dataset, test_dataset = preprocessors.get_and_speed_up_loading()

        pipeline = ImagePipeline(self.config)
        model = pipeline.setup_model()

        # fit model
        history = model.fit(train_dataset,
                            epochs=self.config.NUM_EPOCHS,
                            validation_data=validation_dataset,
                            callbacks=[pipeline.callback])
        # save model using h5

        # Save pipeline to pickle
        pickler = Pickle(self.config)
        pickler.remove_old_pipelines([])
        pickler.save_pipeline(self.pipeline)


if __name__ == "__main__":
    main = ImageMain()
    main.run()
