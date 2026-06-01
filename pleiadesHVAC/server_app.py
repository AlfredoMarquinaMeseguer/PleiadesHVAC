"""baseline: A Flower Baseline."""

import keras
import tensorflow as tf

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pleiadesHVAC.model import load_model
# Create ServerApp
app = ServerApp()


@app.main()
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
    strategy = FedAvg(
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
         
    # Save model in tensorflow
   
    if context.run_config["save-model"]:
        # Save the final model
        ndarrays = result.arrays.to_numpy_ndarrays()
        final_model_name = "final_model.keras"
        print(f"Saving final model to disk as {final_model_name}...")
        model.set_weights(ndarrays)
        model.save(final_model_name)
   
