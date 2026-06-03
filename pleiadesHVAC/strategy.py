"""baseline: A Flower Baseline."""
from typing import Iterable
from logging import INFO, WARNING

import numpy as np

from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import (
    Message,
    MetricRecord,
    ConfigRecord,
    MessageType
)
from flwr.common import (
    ArrayRecord,
    Message,
    MetricRecord,
    RecordDict,
    log,
)

import random
from time import sleep

from typing import cast

from collections.abc import Callable, Iterable

# ---------------------------------------------------------------------------
# Custom Strategy for multiple datasets
# ---------------------------------------------------------------------------

class FedAvgMultiDatasets(FedAvg):
    """
    FedAvg-based strategy with creates a node per available dataset.

    Overrides
    ---------
    __init__  - adds an `available_datasets` argument and creates one node per dataset.
    _construct_messages  - overrides the method to add dataset assignment to the node configuration in each message.
    summary - adds logging of the available datasets and the dataset assignment key.
    """

    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        available_datasets: list[str] = [],
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        datasetrecord_key: str = "dataset_name",
        train_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
    ) -> None:
        
        if len(available_datasets) < 2:
            raise ValueError("At least two datasets must be provided for federated learning.")
        
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=len(available_datasets),
            min_evaluate_nodes=len(available_datasets),
            min_available_nodes=len(available_datasets),
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )
        self.available_datasets = available_datasets        
        self.datasetrecord_key = datasetrecord_key

    # ------------------------------------------------------------------
    # configure_train  (called once per global round, before messages go out)
    # ------------------------------------------------------------------
    def _construct_messages(
        self, record: RecordDict, node_ids: list[int], message_type: str
    ) -> Iterable[Message]:
        """Construct N Messages carrying the same RecordDict payload."""
        messages = []
        for node_id, i in zip(node_ids, range(len(node_ids))):  # one message for each node
            # Overriten part:
            # Configure node-specific options in this case the dataset 
            # assigne to each node
            node_config = record[self.configrecord_key].copy()            
            node_config[self.datasetrecord_key] = self.available_datasets[i] # pyright: ignore[reportArgumentType]
            node_record = RecordDict(
                {
                    self.arrayrecord_key: record[self.arrayrecord_key],
                    self.configrecord_key: node_config,
                }
            )

            message = Message(
                content=node_record,
                message_type=message_type,
                dst_node_id=node_id,
            )
            messages.append(message)
        return messages 
    
    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> Sampling:")
        log(
            INFO,
            "\t│\t├──Fraction: train (%.2f) | evaluate ( %.2f)",
            self.fraction_train,
            self.fraction_evaluate,
        )  # pylint: disable=line-too-long
        log(
            INFO,
            "\t│\t├──Available datasets (1 node/dataset): %s",
            ", ".join(self.available_datasets),
        )  # pylint: disable=line-too-long
        log(INFO, "\t│\t└──Minimum available nodes: %d", self.min_available_nodes)
        log(INFO, "\t└──> Keys in records:")
        log(INFO, "\t\t├── Weighted by: '%s'", self.weighted_by_key)
        log(INFO, "\t\t├── ArrayRecord key: '%s'", self.arrayrecord_key)
        log(INFO, "\t\t└── ConfigRecord key: '%s'", self.configrecord_key)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []
        
        # NOTE: The following part it the overwritten part, where we sample only the exact number of nodes needed
        # and get a warning if not enough nodes are available.
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        
        if num_nodes < self.min_available_nodes:
            log(
                WARNING,
                "Not enough nodes available for training (available: %d, required: %d).",
                len(list(grid.get_node_ids())),
                self.min_available_nodes,
            )
            return []
        
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes)
        # End of overwritten part

        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.TRAIN)
    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        # Do not configure federated evaluation if fraction_evaluate is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # NOTE: The following part it the overwritten part, where we sample only the exact number of nodes needed
        # and get a warning if not enough nodes are available.
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        
        if num_nodes < self.min_available_nodes:
            log(
                WARNING,
                "Not enough nodes available for training (available: %d, required: %d).",
                len(list(grid.get_node_ids())),
                self.min_available_nodes,
            )
            return []
     
        
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes)
        # End of overwritten part
        
        log(
            INFO,
            "configure_evaluate: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)


def sample_nodes(
    grid: Grid, sample_size: int
) -> tuple[list[int], list[int]]:
    """Edit of the Flower strategy_utils function of the same name to wait an exact number of nodes, 
    instead of a max and a min

    Returns
    -------
    tuple[list[int], list[int]]
        A tuple containing the sampled node IDs and the list
        of all connected node IDs.
    """
    sampled_nodes = []

    # wait for min_available_nodes to be online
    while len(all_nodes := list(grid.get_node_ids())) < sample_size:
        log(
            INFO,
            "Waiting for nodes to connect: %d connected (minimum required: %d).",
            len(all_nodes),
            sample_size,
        )
        sleep(1)

    # Sample nodes
    sampled_nodes = random.sample(all_nodes, sample_size)

    return sampled_nodes, all_nodes