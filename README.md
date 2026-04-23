# OPMD Dental Image Classifier Assignment
**School of Mechanical Engineering, University of Leeds**

---

## 🦷 Task Description

Given **900 labelled dental images** across 2 condition classes:
- `cancerous`
- `non_cancerous`

Build a classifier that **maximises macro-averaged F1** on a held-out test set.

---

## 📁 Repository Structure

```
OPMDClass/
├── data/
│   ├── train/
│   │   ├── cancerous/         # Training images - cancerous
│   │   └── non_cancerous/     # Training images - non cancerous
│   └── sample_test/           # Sample test images for self-evaluation
│       ├── cancerous/
│       └── non_cancerous/
├── submission/
│   ├── model.py               # ✏️ YOUR model architecture
│   ├── predict.py             # ✏️ YOUR prediction functions
│   └── requirements.txt       # ✏️ YOUR dependencies
├── evaluate.py                # Instructor grading script
├── sample_evaluate.py         # Self-test script for students
└── .github/
    └── workflows/
        └── validate.yml       # Auto format checker
```

---

## ✅ Submission Checklist

Before the deadline, make sure your repo contains:

- [ ] `submission/model.py` — your model architecture
- [ ] `submission/predict.py` — `load_model()` and `predict()` implemented
- [ ] `submission/best_model.pth` — your saved model weights
- [ ] `submission/requirements.txt` — all dependencies listed
- [ ] `report.pdf` — max 4 pages

---

## 📦 Submission Requirements

### Required Files
| File | Description |
|------|-------------|
| `submission/model.py` | Your model class (must follow the interface) |
| `submission/predict.py` | `load_model()` and `predict()` functions |
| `submission/best_model.pth` | Your trained model weights |
| `submission/requirements.txt` | Python dependencies |
| `report.pdf` | Written report (max 4 pages) |

### Model Interface Rules
- `predict()` must return `0` for `non_cancerous`, `1` for `cancerous`
- `load_model()` must accept a file path string
- Do **not** rename or move `submission/predict.py`

---

## 🎓 Marking Scheme (100 marks total)

| Component | Marks |
|-----------|-------|
| Model Performance (Macro F1) | 40 |
| Code Quality | 20 |
| Written Report | 30 |
| Reproducibility | 10 |

### F1 Performance Breakdown

| Macro F1 Score | Marks |
|----------------|-------|
| ≥ 0.90 | 40 |
| 0.85 – 0.90 | 36 |
| 0.80 – 0.85 | 32 |
| 0.75 – 0.80 | 28 |
| 0.70 – 0.75 | 24 |
| 0.65 – 0.70 | 20 |
| 0.60 – 0.65 | 16 |
| 0.50 – 0.60 | 10 |
| < 0.50 | 5 |

### Late Submission Penalty
| Submission Time | Penalty |
|-----------------|---------|
| < 24 hours late | -10 marks |
| 1–7 days late | -20 marks |
| > 7 days late | Zero |

---

## 🚀 Getting Started

```bash
# Clone your assignment repo
git clone https://github.com/Univ-Leeds-Mech-Engineering/OPMDClass.git
cd OPMDClass

# Install dependencies
pip install -r submission/requirements.txt

# Self-test your model before submitting
python sample_evaluate.py
```

---

## ⏰ Deadline
**TBC by instructor** — set in GitHub Classroom assignment settings.

---

## 📬 Questions?
Use the **GitHub Discussions** tab on this repository.
