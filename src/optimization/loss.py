import numpy as np
import torch
from torch.nn.modules.loss import _Loss
from sklearn.metrics import cohen_kappa_score


class QuadraticWeightedKappa(_Loss):

    def forward(self, predictions, targets):
        return self._quadratic_weighted_kappa(predictions, targets)

    def _quadratic_weighted_kappa(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Gets stuck on the first epoch
        device = predictions.device

        labels = [0, 1, 2, 3, 4]
        kappa = cohen_kappa_score(
            predictions.argmax(dim=1).cpu().numpy().astype(np.uint8),
            targets.cpu().numpy().astype(np.uint8),
            weights="quadratic", labels=labels
        )

        kappa_loss = torch.tensor(1 - kappa)
        kappa_loss.to(device)
        kappa_loss.requires_grad = True

        return kappa_loss
