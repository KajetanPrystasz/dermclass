from dermclass_models.structured.config import StructuredConfig
from dermclass_models.structured.processing.validation import Validation


class Predict:

    def __init__(self, config: StructuredConfig):
        super().__init__(config)

        self.validator = Validation(self.config)
