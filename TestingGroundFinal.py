import torchvision
import ActiveLearning as AL
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Needed for memory expansion
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data_cifar", download=True, train=True)
ac = AL.ActiveLearning(train_dataset, 0.99, 5, 30, delta=0.333, debug=False,quiet=True)
ac.test_methods()

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
ac = AL.ActiveLearning(train_dataset, 0.99, 5, 30, delta=0.778, debug=False,quiet=True)
ac.test_methods()