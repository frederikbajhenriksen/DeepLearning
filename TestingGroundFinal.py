import torchvision
import ActiveLearning as AL
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Needed for memory expansion

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data_cifar", download=True, train=True)
ac = AL.DCoM(train_dataset, unlabelled_size=0.995, label_iterations=7, num_epochs=30, delta=0.515,b=20, debug=False,quiet=True)
ac.test_methods(n_tests=20, increase_b=True, seeds=[37177, 72341, 15706, 20688, 8470, 96544, 5198, 49480, 51022, 63777, 24823, 44543, 19729, 29476, 273, 22255, 39478, 8268, 60422, 67265])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
ac = AL.DCoM(train_dataset, unlabelled_size=0.995, label_iterations=7, num_epochs=30,b=20, delta=0.131, debug=False,quiet=True)
ac.test_methods(n_tests=20, increase_b=True, seeds=[54452, 60252, 515, 2701, 26771, 71796, 73738, 37435, 21004, 18030, 77181, 89046, 93850, 77527, 99708, 59000, 24532, 58560, 2920, 84835])



