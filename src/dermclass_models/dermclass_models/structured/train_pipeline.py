from dermclass_models.structured.config import StructuredConfig
from dermclass_models.structured.processing.pipeline import StructuredPpcPipeline
from dermclass_models.base.processing.preprocessors import Preprocessors


class StructuredMain:

    def __init__(self, config: StructuredConfig):
        super().__init__(config)

        self.preprocessor = Preprocessors(self.config)
        self.ppcpipeline = StructuredPpcPipeline(self.config)


if __name__ == "__main__":
    main = StructuredMain()
    main.run()
