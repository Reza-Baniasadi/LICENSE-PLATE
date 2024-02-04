import torch
# from torchvision import transforms
import albumentations as A
from dataclasses import dataclass
from albumentations.pytorch import ToTensorV2
from alphabets import ALPHABETS
from argparse import Namespace