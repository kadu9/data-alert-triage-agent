import pytest
from pathlib import Path

ZIP = Path(__file__).resolve().parent.parent / "data" / "AI-eng-homeassignment-data.zip"


@pytest.fixture
def real_zip():
    if not ZIP.exists():
        pytest.skip("real dataset zip not present")
    return ZIP
