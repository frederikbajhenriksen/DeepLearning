# DeepLearning
Project 16, for the DTU course 02456 Deep Learning Fall 2024

Implementation of different active learning methods for deeplearning using [resnet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) as the model. The implemented methods are least confidence, margin, entropy and random sampling as the baselines. Testing against [TypiClust](https://arxiv.org/abs/2202.02794), [ProbCover](https://arxiv.org/abs/2205.11320) and [DCoM](https://arxiv.org/abs/2407.01804) implemented from scrath with inspiration from the [original implementations](https://github.com/avihu111/TypiClust). The methods were tested on MNIST and CIFAR10.

The implementation is contained in ActiveLearning.py as a Class with methods for testing and comparisons.