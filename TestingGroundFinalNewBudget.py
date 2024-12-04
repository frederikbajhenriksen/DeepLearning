import torchvision
import ActiveLearning as AL
import os

# Set environment variables for memory handling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define transform for MNIST
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# **Experiment with CIFAR-10**
# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root="./data_cifar", download=True, train=True)

# Create an ActiveLearning instance for CIFAR-10
ac = AL.DCoM(train_dataset, unlabelled_size=0.99, label_iterations=8, num_epochs=30, delta=0.515, b=50, debug=False, quiet=True)

# Define budget schedule
initial_budget = int(0.01 * len(train_dataset))  # Start with 1% of CIFAR-10 dataset
increments = [int(0.02 * len(train_dataset)), int(0.03 * len(train_dataset)), int(0.04 * len(train_dataset)), 
              int(0.05 * len(train_dataset)), int(0.06 * len(train_dataset)), int(0.07 * len(train_dataset)),
              int(0.08 * len(train_dataset))]  # Adjust as needed for 8 iterations

# Set the budget schedule for CIFAR-10
ac.set_budget_schedule(initial_budget, increments)

# Test methods for CIFAR-10
ac.test_methods(n_tests=10)

# **Experiment with MNIST**
# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create an ActiveLearning instance for MNIST
ac = AL.DCoM(train_dataset, unlabelled_size=0.99, label_iterations=8, num_epochs=30, delta=0.131, b=50, debug=False, quiet=True)

# Define budget schedule
initial_budget = int(0.01 * len(train_dataset))  # Start with 1% of MNIST dataset
increments = [int(0.02 * len(train_dataset)), int(0.03 * len(train_dataset)), int(0.04 * len(train_dataset)), 
              int(0.05 * len(train_dataset)), int(0.06 * len(train_dataset)), int(0.07 * len(train_dataset)),
              int(0.08 * len(train_dataset))]  # Adjust as needed for 8 iterations

# Set the budget schedule for MNIST
ac.set_budget_schedule(initial_budget, increments)

# Test methods for MNIST
ac.test_methods(n_tests=10)
