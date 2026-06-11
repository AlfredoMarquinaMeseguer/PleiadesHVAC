"""baseline: A Flower Baseline."""
from typing import Iterable
from logging import INFO, WARNING
from typing import cast
import numpy as np
import os
from collections.abc import Callable, Iterable

from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import (
    Message,
    MetricRecord,
)
from flwr.app import MessageType
from flwr.serverapp.strategy.strategy_utils import sample_nodes
from flwr.common import (
    ArrayRecord,
    Message,
    MetricRecord,
    RecordDict,
    ConfigRecord,
    log,
)
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords



METRICS_FILENAME = "data/metrics/edgefederation_{}.json"

class FedAvgExamples(FedAvg):
    """
    FedAvg-based strategy that saves the global model weights after each round.

    Overrides
    ---------
    _aggregate_fit  - overrides the method to save the global model weights after each round.
    """
    def __init__(
        self,
        ouput_name:str = "",
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
        self.ouput_name = ouput_name

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []
        # Sample nodes
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, self.min_train_nodes)
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
        
    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        # Do not configure federated evaluation if fraction_evaluate is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Sample nodes
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, self.min_evaluate_nodes)
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
 
    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        """Aggregate MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)

        metrics = None
        individual_records = []
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            for record in reply_contents:
                # Get the first (and only) MetricRecord in the record
                individual_records.append(dict(next(iter(record.metric_records.values()))))
        
            # Aggregate MetricRecords
            metrics = self.evaluate_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )

        metrics_file = {
            "individual_metrics" : individual_records,
            "aggregated_metrics" : dict(metrics),
        }

        filepath = METRICS_FILENAME.format(self.ouput_name)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        import json
        with open(filepath, "w") as f:
            f.write(json.dumps(metrics_file, indent=4))
        
        return metrics