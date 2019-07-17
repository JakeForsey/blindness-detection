import numpy as np
import torch
from torch.functional import F
from torch.nn.modules.loss import _Loss
from sklearn.metrics import cohen_kappa_score


class QuadraticWeightedKappa(_Loss):

    def forward(self, predictions, targets):
        # return self._quadratic_weighted_kappa(predictions, targets)
        return self._quadratic_weighted_kappa_2(predictions, targets)

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

    def _quadratic_weighted_kappa_2(self, y1, y2):
        # https://www.kaggle.com/zaharch/minimizing-qwk-directly-with-nn
        batch_size = y1.shape[0]

        numer = (torch.matmul(y2.float().reshape([-1, 1]), torch.tensor(()).new_ones((1, 5)).cuda()) -
                 torch.matmul(torch.tensor(()).new_ones((batch_size, 1)),
                              torch.tensor(range(5), dtype=torch.float).reshape([1, 5])).cuda()) ** 2
        numer = (numer * y1).sum()
        denom = torch.tensor(()).new_ones((batch_size, 1)).cuda()
        denom = (denom * y1).sum()
        loss = numer / denom
        return loss
