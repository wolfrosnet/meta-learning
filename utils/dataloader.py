from typing import *

from torchmeta.datasets.helpers import *
from torchmeta.utils.data import BatchMetaDataLoader


def get_dataloader(
    config: Dict[str, Any]
) -> Tuple[BatchMetaDataLoader, BatchMetaDataLoader, BatchMetaDataLoader]:
    """_summary_

    Args:
        config (Dict[str, Any]): config in main.py

    Returns:
        Tuple[BatchMetaDataLoader, BatchMetaDataLoader, BatchMetaDataLoader]: _description_
    
    available datasets:
    'omniglot',
    'miniimagenet',
    'tieredimagenet',
    'cifar_fs',
    'fc100',
    'cub',
    'doublemnist',
    'triplemnist'
    """
    train_dataset = omniglot(
        folder=config["folder_name"],
        shots=config["num_shots"],
        # test_shots=1, # default = shots
        ways=config["num_ways"],
        shuffle=True,
        meta_train=True,
        download=config["download"],
    )
    train_dataloader = BatchMetaDataLoader(
        train_dataset, batch_size=config["task_batch_size"], shuffle=True, num_workers=1, drop_last=True
    )

    val_dataset = omniglot(
        folder=config["folder_name"],
        shots=config["num_shots"],
        # test_shots=1, # default = shots
        ways=config["num_ways"],
        shuffle=True,
        meta_val=True,
        download=config["download"],
    )
    val_dataloader = BatchMetaDataLoader(
        val_dataset, batch_size=config["task_batch_size"], shuffle=True, num_workers=1,
    )

    test_dataset = omniglot(
        folder=config["folder_name"],
        shots=config["num_shots"],
        # test_shots=1, # default = shots
        ways=config["num_ways"],
        shuffle=True,
        meta_test=True,
        download=config["download"],
    )
    test_dataloader = BatchMetaDataLoader(
        test_dataset, batch_size=config["task_batch_size"], shuffle=True, num_workers=1
    )
    return train_dataloader, val_dataloader, test_dataloader
