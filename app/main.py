print("Hello World!")
import torch
import torchvision
from Model import TrainingVAE, Decoder, Encoder
from Train import BasicTraining
print("preload complete")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device loaded:" + repr(device))
print("Serpent praise")
