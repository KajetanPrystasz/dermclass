from pathlib import Path
import pytest
from dermclass_models2.persistence import BasePersistence


class TestBasePersistence:

    @pytest.mark.integration
    def test_load_pipeline(self):
        pass

    @staticmethod
    def test___remove_files(tmp_path):
        temp_dir = tmp_path / "test_dir"
        temp_dir.mkdir()

        temp_file = temp_dir / "temp_file.txt"
        temp_file.write_text("test")

        temp_subdir = tmp_path / "test_subdir"
        temp_subdir.mkdir()

        BasePersistence._remove_files(temp_dir)
        with pytest.raises(FileNotFoundError):
            list(temp_dir.iterdir())

    def test_remove_old_pipelines(self, tmp_path, testing_config):
        testing_config.PIPELINE_TYPE = "testing_pipeline"
        testing_config.PICKLE_DIR = tmp_path / "test_dir"
        persister = BasePersistence(testing_config)

        temp_dir = tmp_path / "test_dir"
        temp_dir.mkdir()

        temp_file = temp_dir / "temp_file.txt"
        temp_file.write_text("test")

        temp_pipeline = temp_dir / "temp_pipeline"
        temp_pipeline.write_text("test")

        persister.remove_old_pipelines(pipelines_to_keep=[Path("temp_file.txt")])

        with open(str(temp_pipeline), "r") as f:
            f.read()
        with open(str(temp_file), "r") as f:
            f.read()

    @pytest.mark.integration
    def test_save_pipeline(self):
        pass
