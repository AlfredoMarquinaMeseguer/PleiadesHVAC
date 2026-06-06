"""baseline: A Flower Baseline."""
from typing import Iterable
from logging import INFO, WARNING
import io
import os
import numpy as np
import time
from time import sleep
import random
from typing import cast
from collections.abc import Callable, Iterable

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
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

# ---------------------------------------------------------------------------
# Custom Strategy for multiple datasets
# ---------------------------------------------------------------------------
GLOBAL_MODEL_PATH = "state/global_model.npz"
METRICS_FILENAME = "data/metrics/root_federation.json"

class FedAvgMultiDatasets(FedAvg):
    """
    FedAvg-based strategy with creates a node per available dataset.

    Override changes:
    ---------
    __init__  - adds an `available_datasets` argument and creates one node per dataset.
    _construct_messages  - overrides the method to add dataset assignment to the node configuration in each message.
    summary - custom output to refelt the changes to structure
    configure_train - uses the same amount of a nodes as the number of available_datasets
    configure_evaluate - same a configure_train
    start - save global model each round in `GLOBAL_MODEL_PATH` as a numpy array
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
    
     # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result:
        """Execute the federated learning strategy.

        Runs the complete federated learning workflow for the specified number of
        rounds, including training, evaluation, and optional centralized evaluation.

        Parameters
        ----------
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        initial_arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
        num_rounds : int (default: 3)
            Number of federated learning rounds to execute.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_fn : Callable[[int, ArrayRecord], Optional[MetricRecord]], optional
            Optional function for centralized evaluation of the global model. Takes
            server round number and array record, returns a MetricRecord or None. If
            provided, will be called before the first round and after each round.
            Defaults to None.

        Returns
        -------
        Results
            Results containing final model arrays and also training metrics, evaluation
            metrics and global evaluation metrics (if provided) from all rounds.
        """
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        # NOTE: added part is the following two lines to save the global model's inital_arrays 
        os.makedirs(os.path.dirname(GLOBAL_MODEL_PATH), exist_ok=True)
        np.savez_compressed(GLOBAL_MODEL_PATH, *arrays.to_numpy_ndarrays())

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
                # NOTE: the following line overwrites the arrays each round                   
                np.savez_compressed(GLOBAL_MODEL_PATH, *arrays.to_numpy_ndarrays())
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
    
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
        
        os.makedirs(os.path.dirname(METRICS_FILENAME), exist_ok=True)
        import json
        with open(METRICS_FILENAME, "w") as f:
            f.write(json.dumps(metrics_file, indent=4))
        
        return metrics

# NOTE: edited from flwr.serverapp.strategy.strategy to 
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