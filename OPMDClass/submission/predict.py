"""
predict.py — Student Submission File
======================================
Implement load_model() and predict() below.
These functions are called directly by the grading script.

Rules:
  - Do NOT rename this file
  - Do NOT rename load_model() or predict()
  - predict() MUST return an integer: 0 or 1
  - Your saved model file must be named: best_model.pth
"""

import torch
from PIL import Image


# ── Class Mapping (DO NOT CHANGE) ────────────────────────────────
CLASS_LABELS = {
    0: "non_cancerous",
    1: "cancerous"
}
# ─────────────────────────────────────────────────────────────────


def load_model(model_path: str):
    """
    Load and return your trained model ready for inference.

    Args:
        model_path (str): Path to your saved model weights file
                          (always 'submission/best_model.pth')

    Returns:
        model: Your loaded model in eval() mode

    Example:
        from submission.model import DentalClassifier
        model = DentalClassifier()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    """

    model = DentalClassifier()
    model.load_state_dict(torch.load(model_path, map_location = "cpu"))
    model.eval()

    return model
    # ----------------------------------------------------------------
    # TODO: Load your model here
    # ----------------------------------------------------------------


def predict(model, image_path: str) -> int:
    """
    Predict the class of a single dental image.

    Args:
        model:           Your loaded model (returned by load_model)
        image_path (str): Path to a single image file (.jpg / .png)

    Returns:
        int: 0 for non_cancerous, 1 for cancerous

    Example:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1).item()
        return int(pred)
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim = 1).item()

    return int(pred)
    # ----------------------------------------------------------------
    # TODO: Implement your prediction pipeline here
    # ----------------------------------------------------------------
