"""baseline: A Flower Baseline."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import datasets
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from flwr.app import Context
import os


fds: datasets.DatasetDict | None = None  # Cache DatasetDict with pre-split train/test
train_partitioner : IidPartitioner | None = None

def load_data(partition_id, num_partitions, context: Context | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
    # Download and partition dataset
    # Only initialize `DatasetDict` once
    global fds
    global train_partitioner

    if fds is None or train_partitioner is None:
        train_partitioner = IidPartitioner(num_partitions=num_partitions)
        if context is None:
            raise FileNotFoundError("The root directory of the datasets has not been provided.")
        
        dataset_name : str = str(context.run_config["data-set"])
        # Try to load pre-split dataset directory (train.json, test.json)
        dataset_dir = f"data/{dataset_name}"
        
        # Load pre-split DatasetDict
        train_data = Dataset.from_json(f"{dataset_dir}/train.json")
        test_data = Dataset.from_json(f"{dataset_dir}/test.json")
        fds = DatasetDict({"train": train_data, "test": test_data})
            
        train_partitioner.dataset = fds["train"]
    
    train_dataset: datasets.Dataset = train_partitioner.load_partition(partition_id)

    # Get test dataset (not partitioned, same for all clients)
    test_dataset: datasets.Dataset = fds["test"]

    # Format
    train_dataset.set_format(type="numpy", columns=["features", "label"])
    test_dataset.set_format(type="numpy", columns=["features", "label"])

    x_train : np.ndarray = train_dataset[:]["features"].astype("float32")
    y_train : np.ndarray = train_dataset[:]["label"].astype("float32")
    x_test : np.ndarray = test_dataset[:]["features"].astype("float32")
    y_test : np.ndarray = test_dataset[:]["label"].astype("float32")

    return x_train, y_train, x_test, y_test

def get_data_shape(context: Context | None = None):
    global fds

    if fds is None:        
        if context is None:
            raise FileNotFoundError("The root directory of the datasets has not been provided.")
        
        dataset_name : str = str(context.run_config["data-set"])
        # Load pre-split dataset directory (train.json, test.json)
        dataset_dir = f"data/{dataset_name}"

        # Load pre-split DatasetDict
        train_data = Dataset.from_json(f"{dataset_dir}/train.json")
        test_data = Dataset.from_json(f"{dataset_dir}/test.json")
        fds = DatasetDict({"train": train_data, "test": test_data})

    shape = np.array(fds["train"]['features']).shape
    return (shape[1], shape[2])