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
from numpy.typing import NDArray
# Local imports
from .dataset import load_data
from .utils import save_result_to_json
from .strategy import FedAvgExamples
import os

####################################################################
# Aggregator server called by the aggregator_client
# also can be called on its own
####################################################################

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    

# Create ServerApp
server = ServerApp()

GLOBAL_MODEL_PATH = "state/global_model.npz"
RESULTS_OUTPUT_FILE = "state/results/{}_result.json"

@server.main()
def main(grid: Grid, context: Context) -> None:
    """Run entry point for the ServerApp."""
    # Reset local Tensorflow state
    keras.backend.clear_session()    
    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_train = float(context.run_config["fraction-train"])

    # Load global model
    try:
        loaded =  np.load(GLOBAL_MODEL_PATH)
        arrays = [loaded[k] for k in loaded.files]
        arrays = ArrayRecord(arrays)
    except:
        model = load_model(context)
        arrays = ArrayRecord(model.get_weights())

    dataset_name = str(context.run_config["dataset_name"])
    # Initialize FedAvg strategy
    strategy = FedAvgExamples(
        ouput_name=dataset_name,
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
   
    # Export results to JSON
    dataset_name = context.run_config.get("dataset_name", None)
    
    file_path = RESULTS_OUTPUT_FILE.format(dataset_name if dataset_name else "")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    save_result_to_json(result, strategy.num_examples_history[-1], file_path)
