import os
from torchvision import datasets
from torchvision.transforms import ToTensor
def MNIST():
    training_data = datasets.MNIST(
        root=os.environ['DATA_DIR'],
        train=True,
        download=True,
        transform=ToTensor())
    testing_data = datasets.MNIST(
        root=os.environ['DATA_DIR'],
        train=False,
        download=True,
        transform=ToTensor())
    return training_data, testing_data
