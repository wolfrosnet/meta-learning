from typing import List
import datetime as dt

import matplotlib.pyplot as plt


time_now = dt.datetime.now()
run_id = time_now.strftime("%Y%m%d%H%M%S")


def print_graph(
    train_accuracies: List[float],
    val_accuracies: List[float],
    train_losses: List[float],
    val_losses: List[float],
) -> None:
    """_summary_

    Args:
        train_accuracies (List[float]): train accuracies
        val_accuracies (List[float]): validation accuracies
        train_losses (List[float]): train losses
        val_losses (List[float]): validation losses
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].plot(train_accuracies, label="train_acc")
    axes[0].plot(val_accuracies, label="test_acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    
    axes[1].plot(train_losses, label="train_loss")
    axes[1].plot(val_losses, label="test_loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.savefig(f'{run_id}.jpg')

