from typing import *

import torch
import torch.nn as nn
from torchmeta.utils.gradient_based import gradient_update_parameters

from models.models import Conv4


def train_maml(
    device: str,
    task_batch_size: int,
    task_batch: Dict[str, List[torch.Tensor]],
    model: Conv4,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
) -> Tuple[float, float]:
    """_summary_

    Args:
        device (str): cuda or cuda:n
        task_batch_size (int): batch size of task
        task_batch (Dict[str, List[torch.Tensor]]): batch of task data
        model (Conv4): backbone
        criterion (nn.CrossEntropyLoss): loss function
        optimizer (torch.optim.Adam): optimizer

    Returns:
        Tuple[float, float]: accuruacy, outer loss
    """
    model.train()

    support_xs = task_batch["train"][0].to(device=device)
    support_ys = task_batch["train"][1].to(device=device)
    query_xs = task_batch["test"][0].to(device=device)
    query_ys = task_batch["test"][1].to(device=device)

    outer_loss = torch.tensor(0.0, device=device)
    accuracy = torch.tensor(0.0, device=device)

    for support_x, support_y, query_x, query_y in zip(
        support_xs, support_ys, query_xs, query_ys
    ):
        support_prob = model(support_x)
        inner_loss = criterion(support_prob, support_y)

        params = gradient_update_parameters(
            model, inner_loss, step_size=0.4, first_order=False
        )

        query_prob = model(query_x, params=params)
        outer_loss += criterion(query_prob, query_y)

        with torch.no_grad():
            _, query_pred = torch.max(query_prob, dim=-1)
            accuracy += torch.mean(query_pred.eq(query_y).float())

    outer_loss.div_(task_batch_size)

    model.zero_grad()
    outer_loss.backward()
    optimizer.step()

    accuracy.div_(task_batch_size)
    return accuracy.item(), outer_loss.item()


def test_maml(
    device: str,
    task_batch_size: int,
    task_batch: Dict[str, List[torch.Tensor]],
    model: Conv4,
    criterion: nn.CrossEntropyLoss,
) -> Tuple[float, float]:
    model.eval()

    support_xs = task_batch["train"][0].to(device=device)
    support_ys = task_batch["train"][1].to(device=device)
    query_xs = task_batch["test"][0].to(device=device)
    query_ys = task_batch["test"][1].to(device=device)

    outer_loss = torch.tensor(0.0, device=device)
    accuracy = torch.tensor(0.0, device=device)

    for support_x, support_y, query_x, query_y in zip(
        support_xs, support_ys, query_xs, query_ys
    ):
        support_prob = model(support_x)
        inner_loss = criterion(support_prob, support_y)

        params = gradient_update_parameters(
            model, inner_loss, step_size=0.4, first_order=False
        )

        query_prob = model(query_x, params=params)
        outer_loss += criterion(query_prob, query_y)

        with torch.no_grad():
            _, query_pred = torch.max(query_prob, dim=-1)
            accuracy += torch.mean(query_pred.eq(query_y).float())

    outer_loss.div_(task_batch_size)
    accuracy.div_(task_batch_size)
    return accuracy.item(), outer_loss.item()