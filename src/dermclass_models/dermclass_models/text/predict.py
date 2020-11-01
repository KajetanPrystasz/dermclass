from dermclass_models.base.predict import Predict

from dermclass_models.text.config import TextConfig
from dermclass_models.text.processing.validation import Validation


class StructuredPredict(Predict):

    def __init__(self, config: TextConfig = TextConfig):
        super().__init__(config)

        self.validator = Validation(self.config)
