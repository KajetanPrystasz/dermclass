import pandas as pd

from dermclass_models.base.processing.validation import Validation

from dermclass_models.text.config import TextConfig


# TODO: Add text validation
class TextValidation(Validation):

    def __init__(self, config: TextConfig = TextConfig):
        super().__init__(config)

    def custom_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
