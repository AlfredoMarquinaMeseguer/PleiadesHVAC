"""baseline: A Flower Baseline."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from baseline.dataset import load_data
from baseline.model import load_model

import keras
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp



# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""


    # Reset local Tensorflow state
    keras.backend.clear_session()

    # Load the model and initialize it with the received weights
    # Load the model4  
    model = load_model(context.run_config["learning-rate"])
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    x_train, y_train, _, _ = load_data(partition_id, num_partitions)
   

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
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
    model = load_model(context.run_config["learning-rate"])
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    _, _, x_test, y_test = load_data(partition_id, num_partitions)

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
