# Federated learning quantum-classical for image classification 

- **Federated Learning (FL)** is a decentralized machine learning paradigm in which models are trained locally on distributed devices without exchanging raw data, thereby preserving privacy. However, this approach faces several challenges, one of which is the high computational demand. In this project, I propose leveraging the unique capabilities of quantum computing as a potential solution to address this computational burden.
The following hybrid quantum–classical architectural design is employed [paper](https://iopscience.iop.org/article/10.1088/2632-2153/ad2aef).

## 📌 Architectures

### **1. Hybrid Quantum-Classical Model**
- **Description:** A hybrid neural network combining quantum layers with classical convolutional layers for image classification.
- **Architecture Diagram:**  
  ![Architecture Design 1](images/qml%20Design%201.png)
- **Code:** [`hybrid_quantum_classical_models_for_image_classification.ipynb`](hybrid_quantum_classical_models_for_image_classification.ipynb)


---
### **2. FL-based-Hybrid Quantum-Classical Model**
- **Description:** The model in a federated setting with two clients and one server, achieving 98.99% accuracy. This is an initial version of the FL-based implementation  which does not include inter-node communication. The Pennylane-GPU is used.

- **Code:** [`FL-pennygpu-no-mpi.ipynb`](FL-pennygpu-no-mpi.ipynb)


## 🧪 Next Steps
1. **Resolve the MPI-cluster FL quantum-classical code version** to enable distributed training across multiple clients while preserving data privacy and considering the inter-node communication.  
2. **Reproduce the experiments on a real quantum machine** to evaluate performance under realistic quantum hardware constraints.

---




