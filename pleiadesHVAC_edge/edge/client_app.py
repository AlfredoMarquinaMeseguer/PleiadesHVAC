"""baseline: A Flower Baseline."""
import keras
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from matplotlib.style import context
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    max_error,
    median_absolute_error,
)
import os
import json
# Local imports
from .dataset import load_data
from .model import load_model

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Reset local Tensorflow state
    keras.backend.clear_session()

    # Load the data
    dataset_name = context.run_config.get("dataset_name", None)
    if dataset_name is None:
        dataset_name = "buildingA-data"

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    x_train, y_train, _, _ = load_data(partition_id, num_partitions, dataset_name)

    if context.run_config["model-type"] == 'convlstm':
        x_train = np.array(x_train)
        n_samples, timesteps, n_features = x_train.shape
        x_train = x_train.reshape((n_samples, timesteps, 1, n_features, 1))

    # Load the model and initialize it with the received weights
    # Load the model4  
    learning_rate : float = float(context.run_config["learning-rate"])
    model = load_model(context, learning_rate)
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())
    epochs : int = int(context.run_config["local-epochs"])
    batch_size: int = int(context.run_config["batch-size"])
    verbose =  context.run_config.get("verbose")

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    # Get training metrics
    train_loss = history.history["loss"][-1] if "loss" in history.history else None
    train_acc = (
        history.history["accuracy"][-1] if "accuracy" in history.history else None
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.get_weights())
    metrics = {"num-examples": len(x_train)}
    if train_loss is not None:
        metrics["train_loss"] = train_loss
    if train_acc is not None:
        metrics["train_acc"] = train_acc

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Reset local Tensorflow state
    keras.backend.clear_session()

    # Load the model
    model = load_model(context, float(context.run_config["learning-rate"]))
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    # Load the data
    dataset_name = str(context.run_config.get("dataset_name", None))
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    if dataset_name is None:
        dataset_name = "buildingA-data"
    _, _, x_test, y_test = load_data(partition_id, num_partitions, dataset_name)

    # Reshape input for convlstm
    if context.run_config["model-type"] == 'convlstm':
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
    rmse = tf.keras.metrics.RootMeanSquaredError()
    mape = tf.keras.metrics.MeanAbsolutePercentageError()
    msle = tf.keras.metrics.MeanSquaredLogarithmicError()
    cosine_sim = tf.keras.metrics.CosineSimilarity()

    for m in [rmse, mape, msle, cosine_sim]:
        m.update_state(y_test, y_pred)

    # Additional metrics via scikit-learn
    r2_val       = r2_score(y_test, y_pred)
    ev_val       = explained_variance_score(y_test, y_pred)
    max_err_val  = max_error(y_test, y_pred)
    medae_val    = median_absolute_error(y_test, y_pred)

    # NOTE: Can't calculate the relative MAE or RMSE because the data is normalized so 
    # the mean and the standard devation is zero. The same happends for CV(RMSE) and CV(MAE).

    metrics = {
        "mse": eval_loss,
        "mae": eval_acc,
        "medae" : medae_val,
        "rmse" : float(rmse(y_test,y_pred)),
        "msle" : float(msle(y_test, y_pred)),
        "mape" : float(mape(y_test, y_pred)),
        "r2" : r2_val,       
        "explained_variance" : ev_val,
        "max_error" : max_err_val,
        "cosine_similarity": float(cosine_sim(y_test, y_pred)),
        "num-examples": len(x_test),        
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
