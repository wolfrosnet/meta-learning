from typing import *

import torch
import torch.nn as nn
from tqdm import tqdm

from models.models import *
from algo.maml import train_maml, test_maml
from utils.dataloader import get_dataloader
from utils.checkpoint import save_model, load_model
from utils.plot import print_graph


config = {
    "title": "CIFAR-FS 5W5S MAML",
    "folder_name": "dataset",
    "download": True,
    "num_shots": 5,
    "num_ways": 5,
    "output_folder": "saved_model",
    "task_batch_size": 32,
    "num_task_batch_train": 1000,
    "num_task_batch_test": 300,
    "device": "cuda",
    "train": True,
    "test": True,
}


def main():
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(config)
    
    # in_channels=1 for omniglot
    model = Conv4(in_channels=1, out_features=config["num_ways"]).to(device=config["device"])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if config["train"]:
        with tqdm(
            zip(train_dataloader, val_dataloader), total=config["num_task_batch_train"]
        ) as pbar:
            # best_accuracy = torch.tensor(0.0, device=config["device"])
            best_accuracy = 0.0
            train_accuracies = []
            val_accuracies = []
            train_losses = []
            val_losses = []

            for task_batch_idx, (train_batch, val_batch) in enumerate(pbar):
                if task_batch_idx >= config["num_task_batch_train"]:
                    break

                train_accuracy, train_loss = train_maml(
                    device=config["device"],
                    task_batch_size=config["task_batch_size"],
                    task_batch=train_batch,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                )
                val_accuracy, val_loss = test_maml(
                    device=config["device"],
                    task_batch_size=config["task_batch_size"],
                    task_batch=val_batch,
                    model=model,
                    criterion=criterion,
                )

                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                pbar.set_postfix(
                    train_accuracy="{0:.4f}".format(train_accuracy),
                    val_accuracy="{0:.4f}".format(val_accuracy),
                    train_loss="{0:.4f}".format(train_loss),
                    val_loss="{0:.4f}".format(val_loss),
                )
                
                if val_accuracy > best_accuracy:
                    save_model(
                        output_folder=config["output_folder"], 
                        model=model, title="maml_classification.th"
                    )
                    best_accuracy = val_accuracy

        print_graph(
            train_accuracies=train_accuracies,
            val_accuracies=val_accuracies,
            train_losses=train_losses,
            val_losses=val_losses,
        )
    
    if config["test"]:
        load_model(
            output_folder=config["output_folder"], model=model, title="maml_classification.th"
        )

        with tqdm(test_dataloader, total=config["num_task_batch_test"]) as pbar:
            sum_test_accuracies = 0.0
            sum_test_losses = 0.0

            for task_batch_idx, test_batch in enumerate(pbar):
                if task_batch_idx >= config["num_task_batch_test"]:
                    break

                test_accuracy, test_loss = test_maml(
                    device=config["device"],
                    task_batch_size=config["task_batch_size"],
                    task_batch=test_batch,
                    model=model,
                    criterion=criterion,
                )

                sum_test_accuracies += test_accuracy
                sum_test_losses += test_loss
                pbar.set_postfix(
                    test_accuracy="{0:.4f}".format(sum_test_accuracies / (task_batch_idx + 1)),
                    test_loss="{0:.4f}".format(sum_test_losses / (task_batch_idx + 1)),
                )

if __name__=="__main__":
    main()
