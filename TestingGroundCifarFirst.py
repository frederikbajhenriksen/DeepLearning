import torchvision
import ActiveLearning as AL
import os

# Set environment variables for PyTorch CUDA configurations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Needed for memory expansion
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Define transform
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# Analyze CIFAR-10 first
train_dataset_cifar = torchvision.datasets.CIFAR10(root="./data_cifar", download=True, train=True, transform=transform)
ac_cifar = AL.ActiveLearning(train_dataset_cifar, 0.99, 2, 30, debug=False, quiet=True)
ac_cifar.test_methods()

# Then analyze MNIST
train_dataset_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
ac_mnist = AL.ActiveLearning(train_dataset_mnist, 0.99, 2, 30, debug=False, quiet=True)
ac_mnist.test_methods()
