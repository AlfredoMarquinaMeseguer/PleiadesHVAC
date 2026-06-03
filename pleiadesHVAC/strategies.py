"""baseline: A Flower Baseline."""
from typing import Iterable
from logging import INFO, WARNING

import numpy as np

from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import (
    Message,
    MetricRecord,
)
from flwr.common import (
    ArrayRecord,
    Message,
    MetricRecord,
    RecordDict,
    log,
)

from typing import cast

from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

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
        datasetrecord_key: str = "dataset",
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
        for node_id in node_ids:  # one message for each node
            # Overriten part:
            # Configure node-specific options in this case the dataset 
            # assigne to each node
            node_config = record[self.configrecord_key].copy()
            node_config[self.datasetrecord_key] = self.available_datasets[node_id] # pyright: ignore[reportArgumentType]
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


class FedAvgExamples(FedAvg):
    """
    FedAvg-based strategy that saves the global model weights after each round.

    Overrides
    ---------
    _aggregate_fit  - overrides the method to save the global model weights after each round.
    """
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )
        self.num_examples_history = []  # To keep track of the number of examples in each round

    def aggregate_train(
        self,
        server_round: int,    
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate ArrayRecords
            arrays = aggregate_arrayrecords(
                reply_contents,
                self.weighted_by_key,
            )

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )

            # NOTE: This is where we add the saving of the weights 
            # Retrieve weighting factor from MetricRecord
            weights: list[float] = []
            for record in reply_contents:
                # Get the first (and only) MetricRecord in the record
                metricrecord = next(iter(record.metric_records.values()))
                # Because replies have been checked for consistency,
                # we can safely cast the weighting factor to float
                w = cast(float, metricrecord[self.weighted_by_key])
                weights.append(w)

            # Average
            total_weight = sum(weights)
            self.num_examples_history.append(total_weight)

        return arrays, metrics