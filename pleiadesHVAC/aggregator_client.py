import keras
import tensorflow as tf

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.clientapp import ClientApp
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.result import Result
from flwr.cli.run import run


from flwr.simulation import run_simulation, start_simulation

from .model import load_model
import numpy as np
# Local imports
from .dataset import load_data
from .utils import load_result_from_json
from logging import INFO, WARNING
from flwr.common import log

client = ClientApp()

RESULTS_OUTPUT_FILE = "state/results/{}_result.json"
@client.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on local data."""

    dataset_name = str(msg.content["config"]["dataset_name"])    
    model_type: str = str(context.run_config["model-type"])

    run("pleiadesHVAC_edge/", run_config_overrides=[f'dataset_name="{dataset_name}" model-type="{model_type}"']
        ,stream=True)

    file_path = RESULTS_OUTPUT_FILE.format(dataset_name if dataset_name else "")
    result, num_examples = load_result_from_json(file_path)

    train_loss = result.train_metrics_clientapp                \
                .get(len(result.train_metrics_clientapp), {}) \
                .get("train_loss", None)

    train_acc = result.train_metrics_clientapp                \
                .get(len(result.train_metrics_clientapp), {}) \
                .get("train_acc", None)

    model_record = ArrayRecord(result.arrays.to_numpy_ndarrays())
    metrics = { "num-examples" : num_examples }

    if train_loss is not None:
        metrics["train_loss"] = train_loss
    if train_acc is not None:
        metrics["train_acc"] = train_acc

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@client.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the model on local data."""

    # Reset local Tensorflow state
    keras.backend.clear_session()

    # Load the model
    model_type: str = str(context.run_config["model-type"])
    dataset_name = str(msg.content["config"]["dataset_name"])    
    model = load_model(model_type, float(context.run_config["learning-rate"]), dataset_name= dataset_name)
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    dataset_name = str(msg.content["config"]["dataset_name"])
    _, _, x_test, y_test = load_data(partition_id, num_partitions, dataset_name)

    # Reshape input for convlstm
    if context.run_config["model-type"] == 'convlstm':
        x_test = np.array(x_test)
        n_samples, timesteps, n_features = x_test.shape
        x_test = x_test.reshape((n_samples, timesteps, 1, n_features, 1))

    # Evaluate the model
    eval_loss, eval_acc = model.evaluate(x_test, y_test, verbose=0)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(x_test),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
