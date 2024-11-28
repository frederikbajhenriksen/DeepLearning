import torchvision
import oop_version_idx as oop
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Needed for memory expansion



torchvision.datasets.CIFAR10(root="data_cifar", download=True, train=True, transform=torchvision.transforms.ToTensor())




transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)



almodel = oop.ActiveLearning(train_dataset, unlabelled_size=0.99, label_iterations=3, num_epochs=30, b=100, debug=True)
almodel.test_methods(n_tests=10, plot=False, quiet=True)



almodel = oop.ActiveLearning(train_dataset, unlabelled_size=0.99, label_iterations=2, num_epochs=50,debug=False)
almodel.compare_methods()


prob = oop.ProbCover(train_dataset, unlabelled_size=0.99, label_iterations=2, num_epochs=50, debug=False, delta=0.434)



prob.compare_methods()


