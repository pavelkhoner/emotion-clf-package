import pytest

from emotion_model.config.core import ROOT
from emotion_model.processing.data_manager import load_dataset


TEST_IMAGE = ROOT / "tests/test_images/happy (12).png"

@pytest.fixture()
def path():
    return TEST_IMAGE