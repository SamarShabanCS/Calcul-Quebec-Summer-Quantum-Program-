# mpi_fed_qc.py

import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor

import pennylane as qml
from mpi4py import MPI
import numpy as np
import os
# ---------------------------
# PennyLane + PyTorch Model
# ---------------------------
n_qubits = 5
n_layers = 3

# each process creates its own dev
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
if use_cuda is True:
    dev = qml.device("lightning.gpu", wires=n_qubits)
else:
    dev = qml.device("default.qubit", wires=n_qubits)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('dev: ',dev)
#####################

@qml.qnode(dev,interface="torch")
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits)}

# Net arch but with local imports handled
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        # TorchLayer needs the qnode and weight_shapes
        self.qlayer1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer2 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer3 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer4 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc1 = nn.Linear(32 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # split into chunks of 5; ensure size divisible by 5
        x_1, x_2, x_3, x_4 = torch.split(x, 5, dim=1)
        x_1 = self.qlayer1(x_1)
        x_2 = self.qlayer2(x_2)
        x_3 = self.qlayer3(x_3)
        x_4 = self.qlayer4(x_4)
        x = torch.cat([x_1, x_2, x_3, x_4], axis=1)
        x = self.fc3(x)
        return x

# ---------------------------
# MPI + Federated Utilities
# ---------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

server_rank = 0
num_clients = world_size - 1 if world_size > 1 else 1
is_server = (rank == server_rank)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_vector_from_model(model):
    vec = torch.nn.utils.parameters_to_vector([p for p in model.parameters()]).detach().cpu().numpy()
    return vec

def set_model_from_vector(model, vec_np):
    vec = torch.from_numpy(vec_np).to(next(model.parameters()).device)
    torch.nn.utils.vector_to_parameters(vec, [p for p in model.parameters()])

def average_vectors(vecs):
    # vecs: list of numpy arrays
    return np.mean(np.stack(vecs, axis=0), axis=0)

# ---------------------------
# Data partitioning (deterministic)
# ---------------------------
# Assumes `full_dataset` is available in the python path or loaded per process.
# You should replace the dataset loading below with your dataset loader (e.g., torchvision.datasets.MNIST)



# Example dataset: MNIST (greyscale) -- replace with your dataset variable if you already have it

full_dataset = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,
)


# Partition indices among clients (exclude server_rank)
n_total = len(full_dataset)
per_client = n_total // max(1, num_clients)
start = (rank - 1) * per_client if not is_server else None
end = ((rank - 1) * per_client + per_client) if not is_server else None

if not is_server:
    # Last client gets remainder
    if (rank - 1) == num_clients - 1:
        end = n_total
    indices = list(range(start, end))
    subset = Subset(full_dataset, indices)
    train_loader = DataLoader(subset, batch_size=16, shuffle=True)
else:
    # server holds a validation split for evaluation, or you could load a separate test set
    val_dataset = datasets.MNIST(
    root = 'data',
    train = False, 
    download=True,
    transform = ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ---------------------------
# Federated training loop
# ---------------------------
global_model = Net().to(device)
criterion = nn.CrossEntropyLoss()

# If you want server to also participate in training set is_server=False below (we don't)
local_epochs = 1
comm_rounds = 5
lr = 1e-3

# initialize global weights (server broadcast)
global_vector = get_vector_from_model(global_model) if is_server else None
# server broadcasts shape first so clients can allocate same shape
shape = None
if is_server:
    shape = np.array(global_vector.shape, dtype=np.int64)
else:
    shape = np.empty(1, dtype=np.int64)
comm.Bcast(shape, root=server_rank)
shape = tuple(shape.tolist()) if isinstance(shape, np.ndarray) else (shape[0],)

# broadcast initial global model vector
if is_server:
    comm.bcast(global_vector, root=server_rank)
else:
    global_vec = comm.bcast(None, root=server_rank)
    set_model_from_vector(global_model, np.array(global_vec, dtype=np.float32))

for r in range(comm_rounds):
    if is_server:
        print(f"[Server] Starting round {r+1}/{comm_rounds}")
        # gather client vectors
        gathered = comm.gather(None, root=server_rank)
        # gather returns list where clients' entries are numpy arrays; server's entry is None
        client_vecs = [v for v in gathered if v is not None]
        if len(client_vecs) == 0:
            print("[Server] No client updates received!")
            continue
        avg_vec = average_vectors(client_vecs)
        # update global model
        set_model_from_vector(global_model, np.array(avg_vec, dtype=np.float32))
        # broadcast new global vector to all clients
        comm.bcast(avg_vec, root=server_rank)
        # optional: evaluate after aggregation
        global_model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"[Server] Round {r+1} validation accuracy: {100 * correct / total:.2f}%")
    else:
        # CLIENTS
        # receive global model vector (initial or updated)
        recv_vec = comm.bcast(None, root=server_rank)
        set_model_from_vector(global_model, np.array(recv_vec, dtype=np.float32))
        global_model.train()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9)
        # Local training
        for e in range(local_epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = global_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        # send updated parameters to server
        local_vec = get_vector_from_model(global_model).astype(np.float32)
        comm.gather(local_vec, root=server_rank)
        print(f"[Client {rank}] Sent update for round {r+1}")

# Finalize: server already has final global_model
if is_server:
  
    print('CUDA_VISIBLE_DEVICES:  ',os.environ.get("CUDA_VISIBLE_DEVICES"))
    torch.save(global_model.state_dict(), "global_model_final.pt")
    print("[Server] Saved global model to global_model_final.pt")
