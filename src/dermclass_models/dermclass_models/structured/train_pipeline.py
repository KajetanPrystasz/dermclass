from dermclass_models.base.train_pipeline import Main

from dermclass_models.structured.config import StructuredConfig
from dermclass_models.structured.processing.pipeline import StructuredPpcPipeline
from dermclass_models.structured.processing.preprocessors import StructuredPreprocessors


class StructuredMain(Main):

    def __init__(self, config: StructuredConfig = StructuredConfig):
        super().__init__(config)

        self.preprocessor = StructuredPreprocessors(self.config)
        self.ppcpipeline = StructuredPpcPipeline(self.config)


if __name__ == "__main__":
    main = StructuredMain()
    main.run()
