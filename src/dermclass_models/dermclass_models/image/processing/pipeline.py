import logging

import tensorflow as tf

from dermclass_models.image.config import ImageConfig
from dermclass_models.image.processing.preprocessors import ImagePreprocessors


class ImagePipeline:

    def __init__(self, config: ImageConfig = ImageConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.data_augmentation = None
        self.model = None
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    def get_data_augmentation(self, rescale=False):
        layers = [
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1)]

        if rescale:
            layers.append(tf.keras.layers.experimental.preprocessing.Rescaling(1 / 225))

        data_augmentation = tf.keras.Sequential(layers)
        self.data_augmentation = data_augmentation
        return data_augmentation

    def setup_model(self, img_size=None, model_obj=None, learning_rate=None, metrics=None):
        if img_size is None or model_obj is None:
            preprocessor = ImagePreprocessors(self.config)
            img_size, model_obj = preprocessor.get_efficientnet_and_size()

        learning_rate = learning_rate or self.config.LEARNING_RATE
        metrics = metrics or self.config.METRICS

        base_model = model_obj(include_top=False, weights='imagenet', classes=3)
        base_model.trainable = False

        model = tf.keras.Sequential([tf.keras.Input(shape=img_size+(3,)),
                                     self.data_augmentation,
                                     base_model,
                                     tf.keras.layers.GlobalAveragePooling2D,
                                     tf.keras.layers.Dense(len(self.config.DISEASES), "softmax")
                                     ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=metrics)

        self.model = model

        return model
