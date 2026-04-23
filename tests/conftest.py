"""
conftest.py — Shared Test Setup
=================================
Pytest loads this file automatically before running any tests.
It sets up shared fixtures used by both test_interface.py
and test_prediction.py.

Do NOT modify this file.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# ── Paths ─────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent
SUBMISSION_DIR = ROOT_DIR / "submission"
MODEL_PATH     = SUBMISSION_DIR / "best_model.pth"
SAMPLE_TEST    = ROOT_DIR / "data" / "sample_test"
CLASS_LABELS   = {0: "non_cancerous", 1: "cancerous"}
# ──────────────────────────────────────────────────────────────────

# Add repo root to Python path so submission/ can be imported
sys.path.insert(0, str(ROOT_DIR))


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def student_module():
    """
    Load student's predict.py once for the entire test session.
    Fails clearly if the file is missing or has syntax errors.
    """
    predict_path = SUBMISSION_DIR / "predict.py"

    if not predict_path.exists():
        pytest.fail(
            f"submission/predict.py not found at {predict_path}\n"
            "Make sure you have created this file."
        )

    try:
        spec   = importlib.util.spec_from_file_location("predict", predict_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except SyntaxError as e:
        pytest.fail(
            f"submission/predict.py has a syntax error:\n{e}"
        )
    except Exception as e:
        pytest.fail(
            f"Failed to import submission/predict.py:\n{e}"
        )


@pytest.fixture(scope="session")
def loaded_model(student_module):
    """
    Load the student's trained model once for the entire test session.
    Skips prediction tests if best_model.pth is missing.
    """
    if not MODEL_PATH.exists():
        pytest.skip(
            "submission/best_model.pth not found — "
            "save your trained model weights to this path before submitting."
        )

    try:
        model = student_module.load_model(str(MODEL_PATH))
        if model is None:
            pytest.fail("load_model() returned None — make sure it returns your model.")
        return model
    except NotImplementedError:
        pytest.fail(
            "load_model() raises NotImplementedError — "
            "you must implement this function in predict.py."
        )
    except Exception as e:
        pytest.fail(f"load_model() raised an unexpected error:\n{e}")


@pytest.fixture(scope="session")
def sample_images():
    """
    Collect all sample test images with their ground-truth labels.
    Returns list of (image_path_str, label_int) tuples.
    """
    if not SAMPLE_TEST.exists():
        pytest.skip("data/sample_test/ folder not found.")

    samples = []
    for label_idx, label_name in CLASS_LABELS.items():
        class_dir = SAMPLE_TEST / label_name
        if class_dir.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img in sorted(class_dir.glob(ext)):
                    samples.append((str(img), label_idx))

    if not samples:
        pytest.skip(
            "No images found in data/sample_test/ — "
            "add a few sample images to test against."
        )

    return samples
