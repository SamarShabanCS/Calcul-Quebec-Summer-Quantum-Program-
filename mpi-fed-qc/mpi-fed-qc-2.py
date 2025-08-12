import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import pennylane as qml
from collections import OrderedDict
import warnings
import numpy as np
import datetime
from mpi4py import MPI

# Suppress specific UserWarning from torch.nn.init
warnings.filterwarnings("ignore", message="Using a non-full backward hook when the forward contains multiple autograd Nodes")

# Set the device for PyTorch operations
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- PennyLane Quantum Circuit Definition ---
n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch") # IMPORTANT: Specify interface="torch" for PyTorch integration
def qnode(inputs, weights):
    """
    PennyLane quantum circuit for feature embedding and entanglement.
    Inputs are embedded using AngleEmbedding, and then BasicEntanglerLayers
    apply trainable rotations and CNOT gates.
    """
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Define the QLayer weight shapes
n_layers = 3
weight_shapes = {"weights": (n_layers, n_qubits)}

# --- Hybrid Classical-Quantum CNN Architecture ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional Layer 1: processes grayscale images (1 channel)
        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16) # Batch Normalization
        self.pool = nn.MaxPool2d(2, 2) # Max Pooling
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32) # Batch Normalization
        
        # Quantum layers: Four separate PennyLane TorchLayers
        # Each expects 5 inputs, and outputs 5 expectation values (n_qubits)
        self.qlayer1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer2 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer3 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer4 = qml.qnn.TorchLayer(qnode, weight_shapes)
        
        # Fully connected layers (classical part)
        # Input to fc1: 32 channels * 7x7 (after two pooling layers on 28x28 input)
        self.fc1 = nn.Linear(32 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 20) # Output of fc2 is 20, which is split for quantum layers (4 * 5 = 20)
        self.fc3 = nn.Linear(20, 10) # Final output layer for 10 classes (e.g., MNIST digits)

    def forward(self, x):
        # Classical convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 32 * 7 * 7)
        
        # Classical fully connected layers before quantum part
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Split the 20-dimensional output into four 5-dimensional vectors for the four quantum layers
        # This assumes that the output of fc2 is designed to be divisible by n_qubits (5)
        x_1, x_2, x_3, x_4 = torch.split(x, n_qubits, dim=1) 
        
        # Apply quantum layers
        x_1 = self.qlayer1(x_1)
        x_2 = self.qlayer2(x_2)
        x_3 = self.qlayer3(x_3)
        x_4 = self.qlayer4(x_4)
        
        # Concatenate the outputs of the quantum layers
        x = torch.cat([x_1, x_2, x_3, x_4], axis=1)
        
        # Final classical fully connected layer for classification
        x = self.fc3(x)
        return x

# --- Helper functions for parameter conversion ---
def get_model_parameters(net):
    """Returns the current model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_model_parameters(net, parameters):
    """Sets the model parameters from a list of NumPy ndarrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# --- Data Loading and Partitioning for MPI ---
def load_and_partition_data_mpi(rank: int, world_size: int):
    """
    Loads MNIST dataset and partitions it among MPI processes (clients).
    Each client (rank > 0) gets a unique subset of the training and test data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    full_test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Determine the start and end indices for this process's partition
    # for training data
    total_train_size = len(full_train_dataset)
    base_train_size_per_rank = total_train_size // world_size
    start_train_idx = rank * base_train_size_per_rank
    end_train_idx = (rank + 1) * base_train_size_per_rank
    # Last rank takes any remaining data
    if rank == world_size - 1:
        end_train_idx = total_train_size
    
    local_train_indices = list(range(start_train_idx, end_train_idx))
    local_train_dataset = Subset(full_train_dataset, local_train_indices)
    train_loader = DataLoader(local_train_dataset, batch_size=4, shuffle=True)

    # Do the same for validation/test data
    total_test_size = len(full_test_dataset)
    base_test_size_per_rank = total_test_size // world_size
    start_test_idx = rank * base_test_size_per_rank
    end_test_idx = (rank + 1) * base_test_size_per_rank
    if rank == world_size - 1:
        end_test_idx = total_test_size
    
    local_val_indices = list(range(start_test_idx, end_test_idx))
    local_val_dataset = Subset(full_test_dataset, local_val_indices)
    val_loader = DataLoader(local_val_dataset, batch_size=4, shuffle=False)
    
    return train_loader, val_loader

# --- Server (Rank 0) Logic ---
def server_logic(comm, world_size, num_rounds, local_epochs_per_round):
    global_model = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    print(f"Server (Rank 0): Starting federated learning for {num_rounds} rounds.")

    for round_num in range(num_rounds):
        print(f"\nServer (Rank 0): --- Round {round_num + 1}/{num_rounds} ---")
        
        # 1. Send global model parameters to all clients
        global_params = get_model_parameters(global_model)
        for i in range(1, world_size):
            comm.send(global_params, dest=i, tag=11) # Tag 11 for sending params

        # 2. Receive updated parameters and dataset sizes from clients
        client_updates = []
        client_data_sizes = []
        for i in range(1, world_size):
            updated_params, data_size = comm.recv(source=i, tag=22) # Tag 22 for receiving updates
            client_updates.append(updated_params)
            client_data_sizes.append(data_size)
            print(f"Server (Rank 0): Received update from client {i} with {data_size} samples.")

        # 3. Aggregate models (Federated Averaging)
        # Calculate total size of all client datasets
        total_client_data_size = sum(client_data_sizes)
        
        # Initialize averaged parameters with zeros
        averaged_params = [np.zeros_like(p) for p in client_updates[0]]

        # Weighted average based on client data sizes
        for client_idx in range(len(client_updates)):
            weight = client_data_sizes[client_idx] / total_client_data_size
            for param_idx in range(len(averaged_params)):
                averaged_params[param_idx] += weight * client_updates[client_idx][param_idx]
        
        set_model_parameters(global_model, averaged_params)
        print("Server (Rank 0): Aggregated client models.")

        # 4. Evaluate the global model on a central test set (optional)
        # For simplicity, we'll re-use the full test set for server evaluation
        # In a true FL setup, this would be a separate, untainted dataset.
        _, val_loader_server = load_and_partition_data_mpi(0, 1) # Load full test set for server evaluation
        global_model.eval()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for data in val_loader_server:
                images, labels = data
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = global_model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        loss /= len(val_loader_server)
        print(f"Server (Rank 0): Global Model Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

    print("\nServer (Rank 0): Federated learning finished.")

# --- Client (Rank > 0) Logic ---
def client_logic(comm, rank, world_size, num_rounds, local_epochs_per_round):
    local_model = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=0.001, momentum=0.9)

    # Load local data partition
    train_loader, val_loader = load_and_partition_data_mpi(rank, world_size)
    print(f"Client {rank}: Loaded {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples.")

    for round_num in range(num_rounds):
        # 1. Receive global model parameters from the server
        global_params = comm.recv(source=0, tag=11) # Tag 11 for receiving params
        set_model_parameters(local_model, global_params)
        print(f"Client {rank}: Received global model for round {round_num + 1}.")

        # 2. Perform local training
        local_model.train()
        running_loss = 0.0
        start_time = datetime.datetime.now()
        print(f"Client {rank}: Starting local training for {local_epochs_per_round} epochs.")
        for epoch in range(local_epochs_per_round):
            epoch_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Client {rank}, Round {round_num + 1}, Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")
            running_loss += epoch_loss
        end_time = datetime.datetime.now()
        print(f"Client {rank}: Local training finished in {(end_time - start_time).total_seconds():.2f} seconds.")

        # 3. Send updated parameters and local dataset size back to the server
        updated_local_params = get_model_parameters(local_model)
        comm.send((updated_local_params, len(train_loader.dataset)), dest=0, tag=22) # Tag 22 for sending updates
        print(f"Client {rank}: Sent updated model to server.")

    print(f"Client {rank}: Federated learning finished.")

# --- Main MPI Execution Block ---
if __name__ == "__main__":
    comm = MPI.COMM_WORLD # Get the MPI communicator
    world_size = comm.Get_size() # Get the total number of processes
    rank = comm.Get_rank() # Get the rank of the current process


    if world_size < 2:
        print("This script requires at least 2 MPI processes (1 server + at least 1 client).")
        print("Please run with: mpiexec -n <num_processes> python federated_mpi.py")
        exit()

    num_federated_rounds = 5 # Number of global communication rounds
    local_epochs_per_round = 2 # Number of local epochs each client trains for

    if rank == 0:
        server_logic(comm, world_size, num_federated_rounds, local_epochs_per_round)
    else:
        client_logic(comm, rank, world_size, num_federated_rounds, local_epochs_per_round)

