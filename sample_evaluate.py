"""
sample_evaluate.py — Student Self-Test Script
===============================================
Run this script to check your submission works correctly
BEFORE pushing to GitHub.

Usage:
    python sample_evaluate.py

This runs against the sample_test/ images (NOT the hidden test set).
Use your result here as a rough guide only — final marks use
a separate held-out test set.
"""

import sys
import importlib.util
from pathlib import Path
from sklearn.metrics import f1_score, classification_report

# ── Config ───────────────────────────────────────────────────────
SAMPLE_TEST_DIR = Path("data/sample_test")
MODEL_PATH      = Path("submission/best_model.pth")
SUBMISSION_DIR  = Path("submission")
CLASS_LABELS    = {0: "non_cancerous", 1: "cancerous"}
# ─────────────────────────────────────────────────────────────────


def load_student_code():
    spec   = importlib.util.spec_from_file_location(
        "predict", SUBMISSION_DIR / "predict.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_files():
    required = [
        SUBMISSION_DIR / "model.py",
        SUBMISSION_DIR / "predict.py",
        SUBMISSION_DIR / "requirements.txt",
    ]
    print("── Checking required files ──────────────────────")
    all_ok = True
    for f in required:
        if f.exists():
            print(f"  ✅ {f}")
        else:
            print(f"  ❌ MISSING: {f}")
            all_ok = False

    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / 1e6
        print(f"  ✅ {MODEL_PATH}  ({size_mb:.1f} MB)")
    else:
        print(f"  ⚠️  {MODEL_PATH} not found — "
              f"make sure to save your weights here before final submission")

    return all_ok


def run_sample_evaluation():
    print("\n── Running sample evaluation ────────────────────")

    if not MODEL_PATH.exists():
        print("  ⚠️  Skipping inference — best_model.pth not found.")
        print("  Train your model and save weights to submission/best_model.pth")
        return

    if not SAMPLE_TEST_DIR.exists() or not any(SAMPLE_TEST_DIR.rglob("*")):
        print("  ⚠️  No sample test images found in data/sample_test/")
        return

    # Load student code
    try:
        student = load_student_code()
    except Exception as e:
        print(f"  ❌ Could not import predict.py: {e}")
        return

    # Load model
    try:
        model = student.load_model(str(MODEL_PATH))
        print("  ✅ Model loaded")
    except NotImplementedError:
        print("  ❌ load_model() not implemented yet")
        return
    except Exception as e:
        print(f"  ❌ load_model() error: {e}")
        return

    # Gather sample images
    samples = []
    for label_idx, label_name in CLASS_LABELS.items():
        class_dir = SAMPLE_TEST_DIR / label_name
        if class_dir.exists():
            for img in sorted(class_dir.glob("*.jpg")):
                samples.append((str(img), label_idx))
            for img in sorted(class_dir.glob("*.png")):
                samples.append((str(img), label_idx))

    if not samples:
        print("  ⚠️  No images found in data/sample_test/")
        return

    # Run predictions
    y_true, y_pred = [], []
    errors = 0
    for img_path, true_label in samples:
        try:
            pred = student.predict(model, img_path)
            y_true.append(true_label)
            y_pred.append(int(pred))
        except NotImplementedError:
            print("  ❌ predict() not implemented yet")
            return
        except Exception as e:
            errors += 1
            y_true.append(true_label)
            y_pred.append(0)

    # Results
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n  📊 Sample Test Results ({len(samples)} images)")
    print(f"  {'─'*38}")
    print(f"  Macro-averaged F1 : {f1:.4f}")
    print(f"  Prediction errors : {errors}")
    print(f"\n{classification_report(y_true, y_pred, target_names=list(CLASS_LABELS.values()))}")
    print("  ⚠️  These are SAMPLE results. Final marks use a hidden test set.")


if __name__ == "__main__":
    print("=" * 52)
    print("  OPMD Dental Classifier — Submission Checker")
    print("=" * 52)

    files_ok = check_files()
    if not files_ok:
        print("\n❌ Fix missing files before submitting.")
        sys.exit(1)

    run_sample_evaluation()
    print("\n✅ Self-check complete. Push your code to GitHub when ready.")
