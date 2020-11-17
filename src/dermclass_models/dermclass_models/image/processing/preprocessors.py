from typing import Tuple
from pathlib import Path
import logging

import tensorflow as tf
import cv2

from dermclass_models.image.config import ImageConfig


class ImagePreprocessors:

    def __init__(self, config: ImageConfig = ImageConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.model = None

        self.img_size = ()
        self.img_shape = ()

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def get_avg_img_size(self, path: Path = None):
        """ None """
        path = path or self.config.DATA_PATH

        height_list = []
        width_list = []
        for subclass_dir in path.iterdir():
            for img_path in subclass_dir.iterdir():
                img = cv2.imread(str(img_path))
                height, width, _ = img.shape
                height_list.append(height)
                width_list.append(width)
        mean_height = int(sum(height_list) / len(height_list))
        mean_width = int(sum(width_list) / len(width_list))

        self.img_size = (mean_height, mean_width)
        self.img_shape = self.img_size + (3,)

        self.logger.info(f"Mean height is: {mean_height}, mean width is: {mean_width}")

        return self.img_size

    def get_efficientnet_and_size(self, img_size: int = None):
        """https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/"""

        img_size = img_size or self.img_size

        img_size = (img_size[0] + img_size[1]) / 2
        if img_size <= 492:
            img_size = (456, 456)
            model = tf.keras.applications.EfficientNetB5
        elif 492 < img_size <= 564:
            img_size = (528, 528)
            model = tf.keras.applications.EfficientNetB6
        elif 564 < img_size:
            img_size = (600, 600)
            model = tf.keras.applications.EfficientNetB7

        self.img_size = img_size
        self.model = model

        self.logger.info(f"Chosen model={model} with img_size={img_size}")

        return img_size, model

    def get_dataset(self, image_size: Tuple[int, int] = None, batch_size: int = None, data_path: Path = None):
        image_size = image_size or self.img_size
        batch_size = batch_size or self.config.BATCH_SIZE
        data_path = data_path or self.config.DATA_PATH

        train_dataset = (tf.keras.preprocessing
                         .image_dataset_from_directory(data_path,
                                                       validation_split=self.config.TEST_SIZE,
                                                       batch_size=batch_size,
                                                       subset="training",
                                                       seed=self.config.SEED,
                                                       image_size=image_size,
                                                       shuffle=True))
        validation_dataset = (tf.keras.preprocessing
                              .image_dataset_from_directory(data_path,
                                                            validation_split=self.config.TEST_SIZE,
                                                            batch_size=batch_size,
                                                            subset="validation",
                                                            seed=self.config.SEED,
                                                            image_size=image_size,
                                                            shuffle=True))

        validation_batches = tf.data.experimental.cardinality(validation_dataset)
        test_dataset = validation_dataset.take(validation_batches // 2)
        validation_dataset = validation_dataset.skip(validation_batches // 2)

        self.logger.info('Number of train batches: %d' % tf.data.experimental.cardinality(train_dataset))
        self.logger.info('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
        self.logger.info('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        return train_dataset, validation_dataset, test_dataset

    def get_and_speed_up_loading(self, image_size: Tuple[int, int, int] = None, batch_size: int = None, data_path: Path = None):
        image_size = image_size or self.img_size
        batch_size = batch_size or self.config.BATCH_SIZE
        data_path = data_path or self.config.DATA_PATH

        train_dataset, validation_dataset, test_dataset = self.get_dataset(image_size, batch_size, data_path)

        train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        return train_dataset, validation_dataset, test_dataset
