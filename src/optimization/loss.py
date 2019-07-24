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


class Kgbloss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(Kgbloss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()