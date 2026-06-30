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
    num_nodes = int(context.run_config["num-nodes"])
    strategy = FedAvgExamples(
        ouput_name=dataset_name,
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_train_nodes= num_nodes,
        min_available_nodes=num_nodes,
        min_evaluate_nodes= num_nodes,
    )
   
    # Start strategy, run FedAvg for `num_rounds`
    model = load_model(context, float(context.run_config["learning-rate"]))
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(model, str(context.run_config["model-type"])),
    )    
   
    # Export results to JSON
    dataset_name = context.run_config.get("dataset_name", None)
    
    file_path = RESULTS_OUTPUT_FILE.format(dataset_name if dataset_name else "")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    save_result_to_json(result, strategy.num_examples_history[-1], file_path)


   
from flwr.app import Context
from flwr.common import NDArrays, Scalar
from flwr.serverapp import ServerApp

from typing import Dict, Optional, Tuple
import keras
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from matplotlib.style import context
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error,
    median_absolute_error,
)

# Local imports
from .dataset import load_data
from .model import load_model

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    

def get_evaluate_fn(model: tf.keras.Model, model_type:str):
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, model_weigths: ArrayRecord
    ) -> MetricRecord | None:
        """Evaluate the model on local data."""

        # Reset local Tensorflow state
        keras.backend.clear_session()
        # Load the model
        model.set_weights(model_weigths.to_numpy_ndarrays())
        # Load the data
        dataset_name = "combined-data"
        _, _, x_test, y_test = load_data(0, 2, dataset_name)
        
        # Reshape input for convlstm
        if model_type == 'convlstm':
            x_test = np.array(x_test)
            n_samples, timesteps, n_features = x_test.shape
            x_test = x_test.reshape((n_samples, timesteps, 1, n_features, 1))

        # Evaluate the model
        eval_loss, eval_acc = model.evaluate(x_test, y_test, verbose=0)
        
        # Construct and return reply Message
        # Generate predictions
        y_pred = model.predict(x_test, verbose=0).flatten()
        y_test = y_test.flatten()  # ensure 1D

        # Metrics native to TensorFlow/Keras
        mae    = tf.keras.metrics.MeanAbsoluteError()
        mse_m  = tf.keras.metrics.MeanSquaredError()
        rmse_m = tf.keras.metrics.RootMeanSquaredError()
        mape_m = tf.keras.metrics.MeanAbsolutePercentageError()
        msle_m = tf.keras.metrics.MeanSquaredLogarithmicError()
        cosine = tf.keras.metrics.CosineSimilarity()

        for m in [mae, mse_m, rmse_m, mape_m, msle_m, cosine]:
            m.update_state(y_test, y_pred)

        # Additional metrics via scikit-learn
        mae_val      = mean_absolute_error(y_test, y_pred)
        mse_val      = mean_squared_error(y_test, y_pred)
        rmse_val     = np.sqrt(mse_val)
        r2_val       = r2_score(y_test, y_pred)
        mape_val     = mean_absolute_percentage_error(y_test, y_pred) * 100  # as percentage
        ev_val       = explained_variance_score(y_test, y_pred)
        max_err_val  = max_error(y_test, y_pred)
        medae_val    = median_absolute_error(y_test, y_pred)
        msle_val     = mean_squared_error(np.log1p(np.abs(y_test)), np.log1p(np.abs(y_pred)))

        # NOTE: Can't calculate the relative MAE or RMSE because the data is normalized so 
        # the mean and the standard devation is zero. The same happends for CV(RMSE) and CV(MAE).
    
        metrics = {
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "mae" : mae_val,
            "medae" : medae_val,
            "mse" : mse_val,
            "rmse" : rmse_val,
            "msle" : msle_val,
            "mape" : mape_val,
            "r2" : r2_val,       
            "explained_variance" : ev_val,
            "max_error" : max_err_val,
            "cosine_similarity": float(cosine(y_test, y_pred)),
            "num-examples": len(x_test),        
        }

        metric_record = MetricRecord(metrics)

        '''
        metrics_file = dict(metric_record)
        METRICS_FILENAME = "data/metrics/root_federation.json"
        os.makedirs(os.path.dirname(METRICS_FILENAME), exist_ok=True)
        import json
        with open(METRICS_FILENAME, "w") as f:
            f.write(json.dumps(metrics_file, indent=4))
        '''

        return metric_record
        
        
    return evaluate
