import torchvision
import ActiveLearning as AL
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Needed for memory expansion

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])


# train_dataset = torchvision.datasets.CIFAR10(root="./data_cifar", download=True, train=True)
# ac = AL.DCoM(train_dataset, unlabelled_size=0.995, label_iterations=7, num_epochs=30, delta=0.515,b=20, debug=False,quiet=True)
# ac.test_methods(n_tests=20, increase_b=True, seeds=[37177, 72341, 15706, 20688, 8470, 96544, 5198, 49480, 51022, 63777, 24823, 44543, 19729, 29476, 273, 22255, 39478, 8268, 60422, 67265])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
ac = AL.DCoM(train_dataset, unlabelled_size=0.995, label_iterations=7, num_epochs=30,b=25, delta=0.131, debug=False,quiet=True)
ac.test_methods(n_tests=10, increase_b=True,title_append="6")


train_dataset = torchvision.datasets.CIFAR10(root="./data_cifar", download=True, train=True)
ac = AL.DCoM(train_dataset, unlabelled_size=0.999, label_iterations=5, num_epochs=30, delta=0.515,b=25, debug=False,quiet=True)
ac.test_methods(n_tests=20, increase_b=False,title_append="5")

train_dataset = torchvision.datasets.MNIST(root="./data", download=True, train=True, transform=transform)
ac = AL.DCoM(train_dataset, unlabelled_size=0.999, label_iterations=5, num_epochs=30, delta=0.131,b=25, debug=False,quiet=True)
ac.test_methods(n_tests=20, increase_b=False, title_append="5")


# Test 0: {'Random Sampling': {'datapoints': [array([ 269,  294,  344,  444,  644, 1044, 1844])], 'accuracies': [array([[87.51458576],
#        [88.56476079],
#        [90.29838306],
#        [92.03200533],
#        [94.23237206],
#        [95.69928321],
#        [96.68278046]])]}, 'Least Confidence': {'datapoints': [array([ 269,  294,  344,  444,  644, 1044, 1844])], 'accuracies': [array([[88.16469412],
#        [88.59809968],
#        [89.71495249],
#        [90.98183031],
#        [94.63243874],
#        [95.93265544],
#        [94.44907485]])]}, 'Margin Sampling': {'datapoints': [array([ 269,  294,  344,  444,  644, 1044, 1844])], 'accuracies': [array([[88.13135523],
#        [88.79813302],
#        [90.48174696],
#        [91.68194699],
#        [94.24904151],
#        [94.99916653],
#        [94.99916653]])]}, 'Entropy Sampling': {'datapoints': [array([ 269,  294,  344,  444,  644, 1044, 1844])], 'accuracies': [array([[88.44807468],
#        [88.28138023],
#        [89.56492749],
#        [92.54875813],
#        [93.06551092],
#        [94.03233872],
#        [94.78246374]])]}, 'ProbCover': {'datapoints': [array([ 269,  294,  344,  444,  644, 1044, 1844])], 'accuracies': [array([[87.86464411],
#        [88.84814136],
#        [90.19836639],
#        [91.48191365],
#        [91.58193032],
#        [94.18236373],
#        [94.71578596]])]}, 'TypiClust': {'datapoints': [array([ 269,  294,  344,  444,  644, 1044, 1844])], 'accuracies': [array([[88.08134689],
#        [87.64794132],
#        [89.73162194],
#        [88.79813302],
#        [92.49874979],
#        [93.88231372],
#        [93.23220537]])]}, 'DCoM': {'datapoints': [array([ 269,  294,  344,  444,  644, 1044, 1844])], 'accuracies': [array([[88.68144691],
#        [89.06484414],
#        [90.69844974],
#        [91.6486081 ],
#        [93.71561927],
#        [94.63243874],
#        [96.2493749 ]])]}}