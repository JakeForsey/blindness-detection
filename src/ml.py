from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader

from src.optimization.monitoring import APTOSMonitor


def train(model, train_loader, optimizer, device, monitor: Optional[APTOSMonitor] = None):
    model.train()

    if device and torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device('cpu')

    model.to(device)

    losses = []
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        activation = F.log_softmax(output, dim=1)

        loss = F.nll_loss(activation, target)
        loss.backward()

        # One loss per batch
        losses.append(loss)

        optimizer.step()

        monitor.on_train_batch_end(batch_idx, data, train_loader, loss)

    losses = torch.stack(losses)

    monitor.on_train_end(losses, optimizer)


def test(model: torch.nn.Module, test_loader: TorchDataLoader, device, monitor: Optional[APTOSMonitor] = None):
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
    losses = []
    with torch.no_grad():
        for data, target, id_code in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            preds_proba = F.log_softmax(output, dim=1)

            loss = F.nll_loss(preds_proba, target)

            preds = preds_proba.argmax(dim=1)

            predictions_proba.extend(preds_proba.cpu())
            predictions.extend(preds.cpu())
            targets.extend(target)
            ids.extend(id_code)

            # One loss per batch
            losses.append(loss)

            if monitor:
                monitor.on_test_batch_end()

    predictions_proba = torch.stack(predictions_proba)
    predictions = torch.stack(predictions)
    targets = torch.stack(targets)
    losses = torch.stack(losses)

    if monitor:
        monitor.on_test_end(predictions_proba, predictions,  targets, ids, losses)

    return predictions_proba, predictions,  targets, ids, losses


def inference(model: torch.nn.Module, loader: TorchDataLoader, device):
    model.eval()

    if device and torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device('cpu')

    model.to(device)

    predictions = []
    predictions_proba = []
    ids = []
    with torch.no_grad():
        for batch_idx, (data, id_code) in enumerate(loader):
            print(f"Batch {batch_idx}")
            data = data.to(device)

            output = model(data)
            preds_proba = F.log_softmax(output, dim=1)

            preds = preds_proba.argmax(dim=1)

            predictions_proba.extend(preds_proba.cpu())
            predictions.extend(preds.cpu())
            ids.extend(id_code)

    predictions_proba = torch.stack(predictions_proba)
    predictions = torch.stack(predictions)

    return predictions_proba, predictions, ids
