"""
evaluate.py — INSTRUCTOR GRADING SCRIPT
=========================================
Run this against a student's cloned repo to generate
their final mark.

Usage:
    python evaluate.py --student-repo /path/to/student/repo

Requirements (instructor environment):
    pip install scikit-learn Pillow torch torchvision
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ── Config (edit before running) ────────────────────────────────
TEST_DATA_DIR = Path("data/test")          # hidden test set — instructor only
CLASS_LABELS  = {0: "non_cancerous", 1: "cancerous"}
DEADLINE      = "2025-05-01T23:59:59"     # ISO format UTC
# ────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="OPMD Assignment Grader")
    parser.add_argument(
        "--student-repo",
        type=str,
        default=".",
        help="Path to the student's cloned repository",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="marking_result.json",
        help="Path to save marking result JSON",
    )
    return parser.parse_args()


def load_student_code(student_repo: Path):
    predict_path = student_repo / "submission" / "predict.py"
    if not predict_path.exists():
        raise FileNotFoundError(f"predict.py not found at {predict_path}")
    spec   = importlib.util.spec_from_file_location("predict", predict_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(student_repo))
    spec.loader.exec_module(module)
    return module


def get_test_samples(test_dir: Path):
    samples = []
    for label_idx, label_name in CLASS_LABELS.items():
        class_dir = test_dir / label_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Test class folder not found: {class_dir}\n"
                f"Make sure data/test/{label_name}/ exists on the instructor machine."
            )
        for img_path in sorted(class_dir.glob("*.jpg")):
            samples.append((str(img_path), label_idx))
        for img_path in sorted(class_dir.glob("*.png")):
            samples.append((str(img_path), label_idx))
    return samples


def compute_marks(f1_macro: float, is_late: bool, days_late: int) -> dict:
    """Map F1 score to performance marks (out of 40), apply late penalty."""
    thresholds = [
        (0.90, 40), (0.85, 36), (0.80, 32),
        (0.75, 28), (0.70, 24), (0.65, 20),
        (0.60, 16), (0.50, 10),
    ]
    perf = 5
    for threshold, mark in thresholds:
        if f1_macro >= threshold:
            perf = mark
            break

    late_penalty = 0
    if is_late:
        if days_late <= 1:
            late_penalty = 10
        elif days_late <= 7:
            late_penalty = 20
        else:
            late_penalty = perf  # zero performance marks

    return {
        "performance_raw": perf,
        "late_penalty":    late_penalty,
        "performance_net": max(0, perf - late_penalty),
        "code_quality":    0,   # fill in manually after code review
        "report":          0,   # fill in manually after reading report
        "reproducibility": 0,   # fill in manually
        "total_auto":      max(0, perf - late_penalty),
        "note": "Add code_quality (20) + report (30) + reproducibility (10) manually"
    }


def run_evaluation(student_repo: Path, output_path: Path):
    print("=" * 60)
    print("  OPMD DENTAL CLASSIFIER — AUTOMATED MARKING")
    print("=" * 60)
    print(f"  Student repo : {student_repo.resolve()}")
    print(f"  Test data    : {TEST_DATA_DIR.resolve()}")
    print()

    model_path = student_repo / "submission" / "best_model.pth"
    result     = {}

    # ── Late check ───────────────────────────────────────────────
    deadline_dt = datetime.fromisoformat(DEADLINE)
    now_dt      = datetime.utcnow()
    is_late     = now_dt > deadline_dt
    days_late   = max(0, (now_dt - deadline_dt).days) if is_late else 0

    if is_late:
        print(f"  ⚠️  LATE SUBMISSION — {days_late} day(s) after deadline")
    else:
        print("  ✅ Submitted on time")

    # ── Load student code ─────────────────────────────────────────
    try:
        student = load_student_code(student_repo)
        print("  ✅ predict.py imported")
    except Exception as e:
        print(f"  ❌ Failed to import predict.py: {e}")
        result = {"error": str(e), "f1_macro": 0, "marks": compute_marks(0, is_late, days_late)}
        _save(result, output_path)
        return result

    # ── Load model ───────────────────────────────────────────────
    try:
        model = student.load_model(str(model_path))
        print("  ✅ Model loaded")
    except Exception as e:
        print(f"  ❌ load_model() failed: {e}")
        result = {"error": str(e), "f1_macro": 0, "marks": compute_marks(0, is_late, days_late)}
        _save(result, output_path)
        return result

    # ── Run predictions ──────────────────────────────────────────
    try:
        samples = get_test_samples(TEST_DATA_DIR)
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        return

    print(f"\n  Running inference on {len(samples)} test images...")
    y_true, y_pred = [], []
    errors = 0

    for img_path, true_label in samples:
        try:
            pred = student.predict(model, img_path)
            y_true.append(true_label)
            y_pred.append(int(pred))
        except Exception:
            errors += 1
            y_true.append(true_label)
            y_pred.append(0)

    # ── Metrics ──────────────────────────────────────────────────
    f1_macro    = f1_score(y_true, y_pred, average="macro")
    f1_per_cls  = f1_score(y_true, y_pred, average=None)
    accuracy    = accuracy_score(y_true, y_pred)
    cm          = confusion_matrix(y_true, y_pred)
    marks       = compute_marks(f1_macro, is_late, days_late)

    print(f"\n  📊 RESULTS ({len(samples)} images)")
    print(f"  {'─'*40}")
    print(f"  Macro-averaged F1  : {f1_macro:.4f}")
    print(f"  Accuracy           : {accuracy:.4f}")
    print(f"  F1 (non_cancerous) : {f1_per_cls[0]:.4f}")
    print(f"  F1 (cancerous)     : {f1_per_cls[1]:.4f}")
    print(f"  Prediction errors  : {errors}")
    print(f"\n  Confusion Matrix:\n  {cm}")
    print(f"\n{classification_report(y_true, y_pred, target_names=list(CLASS_LABELS.values()))}")
    print(f"  🎓 Performance marks : {marks['performance_net']} / 40")
    if is_late:
        print(f"     (Raw: {marks['performance_raw']}, Late penalty: -{marks['late_penalty']})")
    print(f"  📝 Manually add: code quality (20) + report (30) + reproducibility (10)")

    result = {
        "student_repo"  : str(student_repo.resolve()),
        "timestamp"     : now_dt.isoformat(),
        "is_late"       : is_late,
        "days_late"     : days_late,
        "f1_macro"      : round(f1_macro, 4),
        "accuracy"      : round(accuracy, 4),
        "f1_per_class"  : {
            "non_cancerous": round(float(f1_per_cls[0]), 4),
            "cancerous"    : round(float(f1_per_cls[1]), 4),
        },
        "n_samples"     : len(samples),
        "errors"        : errors,
        "marks"         : marks,
    }

    _save(result, output_path)
    return result


def _save(result, output_path):
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  💾 Result saved to {output_path}")


if __name__ == "__main__":
    args        = parse_args()
    student_dir = Path(args.student_repo)
    output_file = Path(args.output)
    run_evaluation(student_dir, output_file)
