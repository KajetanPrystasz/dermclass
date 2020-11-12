from dermclass_models.base.config import BaseConfig, TestingConfig


class ImageConfig(BaseConfig):

    PIPELINE_TYPE = "image_pipeline"

    VARIABLE_ORDER = ["image", "target"]

    DATA_PATH = BaseConfig.PACKAGE_ROOT / "image" / "datasets"

    IMG_HEIGHT = 255
    IMG_WIDTH = 255

    LEARNING_RATE = 0.001

    METRICS = ["accuracy"]

    BATCH_SIZE = 2

    NUM_EPOCHS = 20

    DISEASES = ["psoriasis", "lichen_planus", "pityriasis_rosea"]


class TestingImageConfig(TestingConfig, ImageConfig):
    PIPELINE_TYPE = "testing_image_pipeline"
