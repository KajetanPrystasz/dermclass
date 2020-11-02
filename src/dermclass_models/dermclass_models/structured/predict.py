from dermclass_models.base.predict import Predict

from dermclass_models.structured.config import StructuredConfig
from dermclass_models.structured.processing.validation import Validation


class StructuredPredict(Predict):

    def __init__(self, config: StructuredConfig = StructuredConfig):
        super().__init__(config)

        self.validator = Validation(self.config)
