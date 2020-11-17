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
        img_size = preprocessors.get_avg_img_size()
        img_size, model = preprocessors.get_efficientnet_and_size(img_size)
        train_dataset, validation_dataset, test_dataset = preprocessors.get_and_speed_up_loading(img_size)

        pipeline = ImagePipeline(self.config)
        data_augmentation = pipeline.get_data_augmentation()
        model = pipeline.setup_model(img_size=img_size, model_obj=model, data_augmentation=data_augmentation)

        # fit model
        history = model.fit(train_dataset,
                            epochs=self.config.NUM_EPOCHS,
                            validation_data=validation_dataset,
                            callbacks=[pipeline.callback])

        # Save pipeline to h5 (pickle)
        pickler = Pickle(self.config)
        pickler.remove_old_pipelines([])
        pickler.save_pipeline(model, to_pck=False)


if __name__ == "__main__":
    main = ImageMain()
    main.run()
