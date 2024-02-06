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
    train_transform = A.Compose(
        [A.Rotate(limit=10, p=0.2),
         A.RandomScale(scale_limit=0.2),
         A.Resize(height=BasicConfig.img_h, width=BasicConfig.img_w),
         A.Normalize(BasicConfig.mean, BasicConfig.std, max_pixel_value=255.0),
         A.ToGray(always_apply=True, p=1),
         ToTensorV2()
         ])
    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Scale()
        transforms.Resize((BasicConfig.img_h, BasicConfig.img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=BasicConfig.mean, std=BasicConfig.std), ]
        )