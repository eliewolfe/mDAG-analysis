import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_special_application(name: str):
    """Import a module from the 'Special Applications' folder (its name contains a space)."""
    path = PROJECT_ROOT / "Special Applications" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def proving_QC_Gaps():
    return load_special_application("proving_QC_Gaps")
