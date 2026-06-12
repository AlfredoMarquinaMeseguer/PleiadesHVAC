"""baseline: A Flower Baseline."""

import os

import keras
import tensorflow as tf

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from .model import load_model
from .strategy import FedAvgMultiDatasets
from flwr.common import log
from logging import INFO
# Create ServerApp
app = ServerApp()

DATASETS_FOLDER = "data/datasets"

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run entry point for the ServerApp."""
    # Reset local Tensorflow state
    keras.backend.clear_session()

    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_train = float(context.run_config["fraction-train"])

    # 
    if not os.path.isdir(DATASETS_FOLDER):
        raise FileNotFoundError("The datasets folder does not exist. Please ensure that the datasets are placed in the correct directory.")


    datasets = [str(os_file) for os_file in os.listdir(DATASETS_FOLDER)]
    datasets.remove('combined-data') # Hotfix to exclude dataset with all the data
    
    # Load global model
    model_type: str = str(context.run_config["model-type"])
    model:tf.keras.Model = load_model(model_type, dataset_name= datasets[0])
    # NOTE: temporal para metricas
    try:
        GLOBAL_MODEL_PATH = "state/global_model.npz"
        loaded =  np.load(GLOBAL_MODEL_PATH)
        arrays = [loaded[k] for k in loaded.files]
        arrays = ArrayRecord(arrays)
    except:
        arrays = ArrayRecord(model.get_weights())
    
    # Initialize FedAvg strategy
    strategy = FedAvgMultiDatasets(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        available_datasets=datasets,
        num_nodes_subfed=[3,7,5],
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn= get_evaluate_fn(model, str(context.run_config["model-type"]))
    )
         
    # Save model in tensorflow
   
    if context.run_config["save-model"]:
        # Save the final model
        ndarrays = result.arrays.to_numpy_ndarrays()        
        final_model_name = "final_model.keras"
        print(f"Saving final model to disk as {final_model_name}...")
        model.set_weights(ndarrays)
        model.save(final_model_name)
   
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

        metrics_file = dict(metric_record)
        METRICS_FILENAME = "data/metrics/root_federation.json"
        os.makedirs(os.path.dirname(METRICS_FILENAME), exist_ok=True)
        import json
        with open(METRICS_FILENAME, "w") as f:
            f.write(json.dumps(metrics_file, indent=4))
            
        return metric_record
        
        
    return evaluate
