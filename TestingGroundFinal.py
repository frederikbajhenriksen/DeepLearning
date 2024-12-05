import torchvision
import ActiveLearning as AL
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Needed for memory expansion

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data_cifar", download=True, train=True)
ac = AL.DCoM(train_dataset, unlabelled_size=0.999, label_iterations=7, num_epochs=30, delta=0.515,b=10, debug=False,quiet=True)
ac.test_methods(n_tests=20, increase_b=True)

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
ac = AL.DCoM(train_dataset, unlabelled_size=0.999, label_iterations=7, num_epochs=30,b=10, delta=0.131, debug=False,quiet=True)
ac.test_methods(n_tests=20, increase_b=True)


