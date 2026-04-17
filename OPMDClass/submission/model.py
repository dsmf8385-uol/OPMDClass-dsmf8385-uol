"""
model.py — Student Submission File
===================================
Define your dental image classifier here.
Your model MUST follow the interface below.

Rules:
  - Class name must be DentalClassifier
  - forward() input shape:  (batch, 3, H, W)  — RGB image tensor
  - forward() output shape: (batch, 2)         — raw logits
  - Do NOT change the class name or method signatures
"""

import torch
import torch.nn as nn


class DentalClassifier(nn.Module):
    """
    Dental image binary classifier.
    Output classes:
        0 = non_cancerous
        1 = cancerous
    """

    def __init__(self):
        super(DentalClassifier, self).__init__()

        # ----------------------------------------------------------------
        # TODO: Define your architecture below
        # You may use CNNs, transfer learning, ViTs, etc.
        #
        # Example (simple CNN — replace with your design):
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )
        # self.classifier = nn.Linear(32 * 112 * 112, 2)
        # ----------------------------------------------------------------
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Logits tensor of shape (batch_size, 2)
        """
        # ----------------------------------------------------------------
        # TODO: Implement your forward pass
        # ----------------------------------------------------------------
        raise NotImplementedError("Implement forward() in model.py")
