"""baseline: A Flower Baseline."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import datasets
import numpy as np
from datasets import load_dataset
from flwr.app import Context


fds: datasets.DatasetDict | None = None  # Cache FederatedDataset
train_partitioner : IidPartitioner | None = None

def load_data(partition_id, num_partitions, context: Context = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    global train_partitioner

    if fds is None or train_partitioner is None:
        train_partitioner = IidPartitioner(num_partitions=num_partitions)
        if context is None:
            raise FileNotFoundError("The root directory of the datasets has not been provided.")
        # Versión con path absoluto
        #data_files = context.run_config["data-path"] + "/" + context.run_config["data-set"]
        data_files = "data/" + context.run_config["data-set"]
        fds = load_dataset("json", data_files=data_files)
        train_partitioner.dataset =  fds["train"]
    
    train_dataset: datasets.Dataset = train_partitioner.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    partition : datasets.DatasetDict = train_dataset.train_test_split(test_size=0.2)

    partition["train"].set_format(type="numpy", columns=["features", "label"])
    partition["test"].set_format(type="numpy", columns=["features", "label"])

    x_train : np.ndarray = partition["train"][:]["features"].astype("float32")
    y_train : np.ndarray = partition["train"][:]["label"].astype("float32")
    x_test : np.ndarray = partition["test"][:]["features"].astype("float32")
    y_test : np.ndarray = partition["test"][:]["label"].astype("float32")

    return x_train, y_train, x_test, y_test

def get_data_shape(context: Context | None = None):
    global fds

    if fds is None:        
        if context is None:
            raise FileNotFoundError("The root directory of the datasets has not been provided.")
        # Versión con path absoluto
        #data_files = context.run_config["data-path"] + "/" + context.run_config["data-set"]
        data_files = "data/" + context.run_config["data-set"]
        fds = load_dataset("json", data_files=data_files)        

    shape = np.array(fds["train"]['features']).shape
    return (shape[1], shape[2])