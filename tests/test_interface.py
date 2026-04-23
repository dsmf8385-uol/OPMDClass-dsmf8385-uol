"""
test_interface.py — Code Structure & Interface Tests
======================================================
These tests check that the student's submission follows
the required interface WITHOUT needing model weights.

✅ Safe to run on GitHub Actions on every push.
✅ Students see these results immediately after pushing.

Run:
    pytest tests/test_interface.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent
SUBMISSION_DIR = ROOT_DIR / "submission"
MODEL_PATH     = SUBMISSION_DIR / "best_model.pth"
# ──────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════
# 1. FILE EXISTENCE CHECKS
# ══════════════════════════════════════════════════════════════════

class TestRequiredFiles:
    """Check all required submission files are present."""

    def test_model_py_exists(self):
        """submission/model.py must exist."""
        path = SUBMISSION_DIR / "model.py"
        assert path.exists(), (
            "❌ submission/model.py is missing.\n"
            "Create your model architecture in this file."
        )

    def test_predict_py_exists(self):
        """submission/predict.py must exist."""
        path = SUBMISSION_DIR / "predict.py"
        assert path.exists(), (
            "❌ submission/predict.py is missing.\n"
            "Implement load_model() and predict() in this file."
        )

    def test_requirements_txt_exists(self):
        """submission/requirements.txt must exist."""
        path = SUBMISSION_DIR / "requirements.txt"
        assert path.exists(), (
            "❌ submission/requirements.txt is missing.\n"
            "List all your Python dependencies in this file."
        )

    def test_requirements_not_empty(self):
        """requirements.txt must not be empty."""
        path = SUBMISSION_DIR / "requirements.txt"
        if not path.exists():
            pytest.skip("requirements.txt missing — checked in another test.")
        content = path.read_text().strip()
        assert content, (
            "❌ submission/requirements.txt is empty.\n"
            "List your dependencies, e.g. torch, torchvision, Pillow."
        )

    def test_best_model_pth_exists(self):
        """submission/best_model.pth must exist for final submission."""
        assert MODEL_PATH.exists(), (
            "⚠️  submission/best_model.pth is missing.\n"
            "Train your model and save weights using:\n"
            "    torch.save(model.state_dict(), 'submission/best_model.pth')"
        )


# ══════════════════════════════════════════════════════════════════
# 2. MODEL CLASS CHECKS
# ══════════════════════════════════════════════════════════════════

class TestModelClass:
    """Check DentalClassifier class structure in model.py."""

    @pytest.fixture(autouse=True)
    def import_model(self):
        """Import DentalClassifier for all tests in this class."""
        model_path = SUBMISSION_DIR / "model.py"
        if not model_path.exists():
            pytest.skip("model.py missing — checked in TestRequiredFiles.")
        try:
            sys.path.insert(0, str(ROOT_DIR))
            from submission.model import DentalClassifier
            self.DentalClassifier = DentalClassifier
        except SyntaxError as e:
            pytest.fail(f"❌ model.py has a syntax error:\n{e}")
        except ImportError as e:
            pytest.fail(f"❌ Could not import DentalClassifier from model.py:\n{e}")

    def test_dental_classifier_class_exists(self):
        """DentalClassifier class must exist in model.py."""
        assert self.DentalClassifier is not None, (
            "❌ DentalClassifier class not found in submission/model.py.\n"
            "Make sure the class is named exactly DentalClassifier."
        )

    def test_inherits_nn_module(self):
        """DentalClassifier must extend torch.nn.Module."""
        assert issubclass(self.DentalClassifier, nn.Module), (
            "❌ DentalClassifier must extend torch.nn.Module.\n"
            "Define your class as: class DentalClassifier(nn.Module):"
        )

    def test_has_forward_method(self):
        """DentalClassifier must implement forward()."""
        assert hasattr(self.DentalClassifier, "forward"), (
            "❌ DentalClassifier must implement a forward() method."
        )

    def test_can_instantiate(self):
        """DentalClassifier must be instantiable with no arguments."""
        try:
            model = self.DentalClassifier()
            assert model is not None
        except NotImplementedError:
            pytest.skip("Model not yet implemented — skipping instantiation test.")
        except Exception as e:
            pytest.fail(
                f"❌ Could not instantiate DentalClassifier():\n{e}\n"
                "Make sure __init__() takes no required arguments."
            )

    def test_forward_output_shape(self):
        """forward() must return tensor of shape (batch_size, 2)."""
        try:
            model  = self.DentalClassifier()
            dummy  = torch.randn(4, 3, 224, 224)   # batch=4, RGB, 224x224
            output = model(dummy)
            assert output.shape == (4, 2), (
                f"❌ forward() output shape is {tuple(output.shape)}, "
                f"expected (4, 2).\n"
                f"Your model must output 2 logits per image (one per class)."
            )
        except NotImplementedError:
            pytest.skip("forward() not yet implemented.")
        except Exception as e:
            pytest.fail(f"❌ forward() raised an error:\n{e}")

    def test_forward_output_is_tensor(self):
        """forward() must return a torch.Tensor."""
        try:
            model  = self.DentalClassifier()
            dummy  = torch.randn(2, 3, 224, 224)
            output = model(dummy)
            assert isinstance(output, torch.Tensor), (
                f"❌ forward() must return a torch.Tensor, got {type(output)}."
            )
        except NotImplementedError:
            pytest.skip("forward() not yet implemented.")
        except Exception as e:
            pytest.fail(f"❌ forward() raised an error:\n{e}")


# ══════════════════════════════════════════════════════════════════
# 3. PREDICT.PY INTERFACE CHECKS
# ══════════════════════════════════════════════════════════════════

class TestPredictInterface:
    """Check predict.py has all required functions and constants."""

    def test_load_model_function_exists(self, student_module):
        """predict.py must define load_model()."""
        assert hasattr(student_module, "load_model"), (
            "❌ load_model() not found in submission/predict.py.\n"
            "Define: def load_model(model_path: str):"
        )

    def test_load_model_is_callable(self, student_module):
        """load_model must be a callable function."""
        assert callable(student_module.load_model), (
            "❌ load_model in predict.py is not a function."
        )

    def test_predict_function_exists(self, student_module):
        """predict.py must define predict()."""
        assert hasattr(student_module, "predict"), (
            "❌ predict() not found in submission/predict.py.\n"
            "Define: def predict(model, image_path: str) -> int:"
        )

    def test_predict_is_callable(self, student_module):
        """predict must be a callable function."""
        assert callable(student_module.predict), (
            "❌ predict in predict.py is not a function."
        )

    def test_class_labels_exists(self, student_module):
        """CLASS_LABELS must be defined in predict.py."""
        assert hasattr(student_module, "CLASS_LABELS"), (
            "❌ CLASS_LABELS not found in submission/predict.py.\n"
            "Do not remove CLASS_LABELS from predict.py."
        )

    def test_class_labels_not_modified(self, student_module):
        """CLASS_LABELS must remain {0: 'non_cancerous', 1: 'cancerous'}."""
        expected = {0: "non_cancerous", 1: "cancerous"}
        actual   = student_module.CLASS_LABELS
        assert actual == expected, (
            f"❌ CLASS_LABELS was modified.\n"
            f"Expected : {expected}\n"
            f"Got      : {actual}\n"
            f"Do NOT change CLASS_LABELS in predict.py."
        )

    def test_load_model_signature(self, student_module):
        """load_model() must accept exactly one argument (model_path)."""
        import inspect
        sig    = inspect.signature(student_module.load_model)
        params = list(sig.parameters.keys())
        assert len(params) == 1, (
            f"❌ load_model() must accept exactly 1 argument (model_path).\n"
            f"Found arguments: {params}"
        )

    def test_predict_signature(self, student_module):
        """predict() must accept exactly two arguments (model, image_path)."""
        import inspect
        sig    = inspect.signature(student_module.predict)
        params = list(sig.parameters.keys())
        assert len(params) == 2, (
            f"❌ predict() must accept exactly 2 arguments (model, image_path).\n"
            f"Found arguments: {params}"
        )
