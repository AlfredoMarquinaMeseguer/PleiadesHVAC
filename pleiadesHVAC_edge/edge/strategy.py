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