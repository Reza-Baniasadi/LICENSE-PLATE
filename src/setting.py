import torch
#from torchvision import transforms
import albumentations as A
from dataclasses import dataclass
from albumentations.pytorch import ToTensorV2
from alphabets import ALPHABETS
from argparse import Namespace


@dataclass(init=True)
class BasicConfig:
    img_h = 32 
    img_w = 100  

    file_name = "best"

    # Modify
    n_classes = 35
    mean = [0.4845]
    std = [0.1884]
    alphabet_name = "FA_LPR"
    train_directory = '/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/train'
    val_directory = '/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/val'
    output_dir = "output"

    def update_basic(self):
        self.n_classes = len(self.alphabets) + 1


@dataclass(init=True, repr=True)
class AugConfig(BasicConfig):
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((BasicConfig.img_h, BasicConfig.img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=BasicConfig.mean, std=BasicConfig.std), ]
    )
    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Scale()
        transforms.Resize((BasicConfig.img_h, BasicConfig.img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=BasicConfig.mean, std=BasicConfig.std), ]
    )