import keras
import tensorflow as tf

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.clientapp import ClientApp
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.result import Result


from flwr.simulation import run_simulation

from .model import load_model
import numpy as np
# Local imports
from .dataset import load_data
from .utils import save_result_to_json
from .strategy import FedAvgExamples

####################################################################
# Aggregator composed of a ClientApp and a ServerApp
# ClientApp runs a simulation where the serverApp is that of the aggergator 
# and the clientApp is that of the normal edge nodes that train on data.
####################################################################

# Create ServerApp
server = ServerApp()

@server.main()
def main(grid: Grid, context: Context) -> None:
    """Run entry point for the ServerApp."""
    # Reset local Tensorflow state
    keras.backend.clear_session()    
    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_train = float(context.run_config["fraction-train"])

    # Load global model
    model:tf.keras.Model = load_model(context=context)
    arrays = ArrayRecord(model.get_weights())

    # Initialize FedAvg strategy
    strategy = FedAvgExamples(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_available_nodes=2,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )    

    dataset_name = context.run_config.get("dataset_name", None)

    results_filename = f"{dataset_name}_result.json" if dataset_name else "result.json"    
    save_result_to_json(result, strategy.num_examples_history[-1], results_filename)
