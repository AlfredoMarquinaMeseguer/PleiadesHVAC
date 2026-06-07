"""baseline: A Flower Baseline."""
import os

import keras
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers
from .models import GRUModelBuilder, GRUSimpleModelBuilder, ConvLSTMModelBuilder, LSTMModelBuilder, TransformerModelBuilder
from pleiadesHVAC.dataset import get_data_shape
from flwr.app import Context

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_model(model_type: str, learning_rate: float = 0.001, dataset_name:str = "buildingA-data"):
    
    input_shape = get_data_shape(dataset_name)
    model = None
    match model_type.casefold():
        case "convlstm":
            input_shape = (input_shape[0], 1, input_shape[1], 1)
            model = ConvLSTMModelBuilder(input_shape=input_shape, learning_rate=learning_rate)
        case "lstm":    
            model = LSTMModelBuilder(input_shape=input_shape, learning_rate= learning_rate)
        case "gru":    
            model = GRUModelBuilder(input_shape=input_shape, learning_rate=learning_rate)
        case "grusimple":    
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