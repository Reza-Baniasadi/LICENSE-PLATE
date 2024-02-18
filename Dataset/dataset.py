import os
from argparse import ArgumentParser
from os.path import join
from os.path import split

import albumentations
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from deep_utils import split_extension, log_print
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np


class CRNNDataset(Dataset):

    def __init__(self, root, characters, transform=None, logger=None):
        self.transform = transform
        self.char2label = {char: i + 1 for i, char in enumerate(characters)}
        self.label2char = {label: char for char, label in self.char2label.items()}
        self.image_paths, self.labels, self.labels_length = self.get_image_paths(root, characters,
                                                                                 chars2label=self.char2label,
                                                                                 logger=logger)
        self.n_classes = len(self.label2char) + 1  