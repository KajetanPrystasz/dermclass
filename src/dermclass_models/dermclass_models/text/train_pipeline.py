from dermclass_models.base.train_pipeline import Main

from dermclass_models.text.config import TextConfig
from dermclass_models.text.processing.pipeline import TextPpcPipeline
from dermclass_models.text.processing.preprocessors import TextPreprocessors


class TextMain(Main):

    def __init__(self, config: TextConfig = TextConfig):
        super().__init__(config)

        self.preprocessor = TextPreprocessors(self.config)
        self.ppcpipeline = TextPpcPipeline(self.config)


if __name__ == "__main__":
    main = TextMain()
    main.run()
