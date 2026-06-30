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

DATASETS_DIR = "data/datasets"

def load_data(partition_id, num_partitions, dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
    # Download and partition dataset
    # Only initialize `DatasetDict` once
    global fds
    global train_partitioner

    if fds is None or train_partitioner is None:
        if not os.path.exists(DATASETS_DIR):    
            raise FileNotFoundError(f"The root directory of the datasets has not been provided. Expected directory: {DATASETS_DIR}")
        
        dataset_dir = f"{DATASETS_DIR}/{dataset_name}"

        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        # Try to load pre-split dataset directory (train.json, test.json)        
        # Load pre-split DatasetDict
        train_data = Dataset.from_json(f"{dataset_dir}/train.json")
        test_data = Dataset.from_json(f"{dataset_dir}/test.json")
        fds = DatasetDict({"train": train_data, "test": test_data})

        train_partitioner = IidPartitioner(num_partitions=num_partitions)
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

def get_data_shape(dataset_name: str | None = None):
    global fds

    if fds is None:        
        if dataset_name is None:
            raise FileNotFoundError("The dataset directory has not been provided.")        
        
        # Load pre-split dataset directory (train.json, test.json)
        dataset_dir = f"{DATASETS_DIR}/{dataset_name}"

        # Load pre-split DatasetDict
        train_data = Dataset.from_json(f"{dataset_dir}/train.json")
        test_data = Dataset.from_json(f"{dataset_dir}/test.json")
        fds = DatasetDict({"train": train_data, "test": test_data})

    shape = np.array(fds["train"]['features']).shape
    return (shape[1], shape[2])