from typing import Tuple
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader


def train(log_interval, model, train_loader, optimizer, epoch, device):
    model.train()

    if device and torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device('cpu')

    model.to(device)

    for batch_idx, (data, target, _) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        activation = F.log_softmax(output, dim=1)

        loss = F.nll_loss(activation, target)
        loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model: torch.nn.Module, test_loader: TorchDataLoader, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    model.eval()

    if device and torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device('cpu')

    model.to(device)

    predictions = []
    predictions_proba = []
    targets = []
    ids = []
    with torch.no_grad():
        for data, target, id_code in test_loader:
            data = data.to(device)

            output = model(data)
            preds_proba = F.log_softmax(output, dim=1)

            preds = preds_proba.argmax(dim=1)

            predictions_proba.extend(preds_proba.cpu())
            predictions.extend(preds.cpu())
            targets.extend(target)
            ids.extend(id_code)

    # TODO create a TestResult data class to contaion these results
    return torch.stack(predictions_proba), torch.stack(predictions),  torch.stack(targets), ids
