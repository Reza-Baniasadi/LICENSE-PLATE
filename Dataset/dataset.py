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
        self.image_paths, self.labels, self.labels_length = self.get_image_paths(root, characters,chars2label=self.char2label,logger=logger)
        self.n_classes = len(self.label2char) + 1  

    @staticmethod
    def text2label(char2label: dict, text: str):
        return [char2label[t] for t in text]
    
    @staticmethod
    def get_image_paths(root, chars, chars2label, logger=None):
        paths, labels, labels_length = [], [], []
        discards = 0
        for img_name in os.listdir(root):
            img_path = join(root, img_name)
            try:
                if split_extension(img_name)[-1].lower() in ['.jpg', '.png', '.jpeg']:
                    text = CRNNDataset.get_label(img_path)
                    is_valid, character = CRNNDataset.check_validity(text, chars)
                    if is_valid:
                        label = CRNNDataset.text2label(chars2label, text)
                        labels.append(label)
                        paths.append(img_path)
                        labels_length.append(len(label))
                    else:
                        log_print(logger,
                                  f"[Warning] text for sample: {img_path} is invalid because of the following character: {character}")
                        discards += 1
                else:
                    log_print(logger, f"[Warning] sample: {img_path} does not have a valid extension. Skipping...")
                    discards += 1
            except:
                log_print(logger, f"[Warning] sample: {img_path} is not valid. Skipping...")
                discards += 1
        assert len(labels) == len(paths)
        log_print(logger, f"Successfully gathered {len(labels)} samples and discarded {discards} samples!")

        return paths, labels, labels_length
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.image_paths[index]
        if isinstance(self.transform, albumentations.core.composition.Compose):
            #img = cv2.imread(img_path)[..., ::-1]  
            img = np.array(Image.open(img_path))[..., :3]
            img = self.transform(image=img)['image'][0:1, ...].unsqueeze(0)  
        else:
            img = Image.fromarray(np.array(Image.open(img_path))[..., :3])
            img = self.transform(img).unsqueeze(0)  
        label = torch.LongTensor(self.labels[index]).unsqueeze(0)
        label_length = torch.LongTensor([self.labels_length[index]]).unsqueeze(0)

        return img, label, label_length
    
    @staticmethod
    def check_validity(text, chars):
        for c in text:
            if c not in chars:
                return False, c
        return True, None

    @staticmethod
    def collate_fn(batch):
        images, labels, labels_lengths = zip(*batch)
        images = torch.cat(images, dim=0)
        labels = [label.squeeze(0) for label in labels]
        labels = nn.utils.rnn.pad_sequence(labels, padding_value=-100).T
        labels_lengths = torch.cat(labels_lengths, dim=0)
        return images, labels, labels_lengths
    
def get_mean_std(dataset_dir, alphabets, batch_size, img_h, img_w):

    transformations = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor()]
    )

    dataset = CRNNDataset(root=dataset_dir, transform=transformations, characters=alphabets)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    mean, std = 0, 0
    n_samples = len(dataset)
    for images, labels, labels_lengths in tqdm(data_loader, desc="Getting mean and std"):
        # channel wise
        mean += torch.sum(torch.mean(images, dim=(2, 3)), dim=0)
        std += torch.sum(torch.std(images, dim=(2, 3)), dim=0)
    mean /= n_samples
    std /= n_samples
    return [round(m, 4) for m in mean.numpy().tolist()], [round(s, 4) for s in std.numpy().tolist()]
