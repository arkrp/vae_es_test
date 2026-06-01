print("Hello World!")
import torch
import torchvision
from Datasets import MNIST
print("preload complete")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device loaded:" + repr(device))

print("Serpent praise")
