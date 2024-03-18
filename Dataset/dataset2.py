import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class OCRDataset(Dataset):
    def __init__(self, root_dir, alphabets, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.alphabets = alphabets
        self.img_paths = []
        self.labels = []

        for file_name in os.listdir(root_dir):
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                label = os.path.splitext(file_name)[0] 
                self.img_paths.append(os.path.join(root_dir, file_name))
                self.labels.append(label)

        self.char_to_idx = {c: i + 1 for i, c in enumerate(alphabets)} 

    def __len__(self):
        return len(self.img_paths)

    def encode_label(self, text):
        return [self.char_to_idx[c] for c in text]

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_text = self.labels[idx]
        label_encoded = torch.tensor(self.encode_label(label_text), dtype=torch.long)

        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)

        return img, label_encoded
