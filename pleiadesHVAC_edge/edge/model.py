"""baseline: A Flower Baseline."""
import os

import keras
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers
from .models import GRUModelBuilder, GRUSimpleModelBuilder, ConvLSTMModelBuilder, LSTMModelBuilder, TransformerModelBuilder
from .dataset import get_data_shape
from flwr.app import Context

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

'''
def load_model(learning_rate: float = 0.001):
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
'''

def load_model(context, learning_rate: float = 0.001):
    dataset_name = context.run_config["dataset_name"]
    model_type = context.run_config["model-type"]

    input_shape = get_data_shape(dataset_name)
    model = None    
    match model_type:
        case "convlstm":
            input_shape = (input_shape[0], 1, input_shape[1], 1)
            model = ConvLSTMModelBuilder(input_shape=input_shape, learning_rate=learning_rate)
        case "lstm":    
            model = LSTMModelBuilder(input_shape=input_shape, learning_rate= learning_rate)
        case "gru":    
            model = GRUModelBuilder(input_shape=input_shape, learning_rate=learning_rate)
        case "gruSimple":    
            model = GRUSimpleModelBuilder(input_shape=input_shape, learning_rate=learning_rate)
        case "transformer":    
            model = TransformerModelBuilder(input_shape=input_shape, learning_rate=learning_rate)
        case _:
            raise ValueError("Model type selected in configuration not an allowed model. Please make sure the value entered "+
                         "either through console or configuration is one of the following values: transformer, lstm, "+
                         "convlstm, gru or gruSimple.\n You can find the value inside pyproject.toml, section "+
                         "[tool.flwr.app.config] and be called 'model-type'")        
    
    # TODO: cambiar para que la se seleccione el tipo de modelo segun config
    return model.build()

'''
def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
'''