"""baseline: A Flower Baseline."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

fds = None  # Cache FederatedDataset

def load_data(partition_id, num_partitions) -> tuple[any, any, any, any] :
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)

    partition["train"].set_format(type="numpy", columns=["img", "label"])
    partition["test"].set_format(type="numpy", columns=["img", "label"])

    x_train = partition["train"][:]["img"].astype("float32") / 255.0
    y_train = partition["train"][:]["label"]
    x_test = partition["test"][:]["img"].astype("float32") / 255.0
    y_test = partition["test"][:]["label"]

    return x_train, y_train, x_test, y_test