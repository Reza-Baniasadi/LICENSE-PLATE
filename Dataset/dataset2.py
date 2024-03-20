import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class DictOCRDataset(Dataset):
    def __init__(self, folder_path, alphabet, transform=None):
        self.transform = transform
        self.vocab = alphabet
        self.data = []

        self.char_to_idx = {c: i + 1 for i, c in enumerate(alphabet)}

        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".png")):
                label_text = os.path.splitext(fname)[0]
                self.data.append({"img_path": os.path.join(folder_path, fname),
                                  "label_text": label_text})

    def text_to_indices(self, text):
        return [self.char_to_idx[c] for c in text]

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = Image.open(sample["img_path"]).convert("L")
        if self.transform:
            img = self.transform(img)

        label_encoded = torch.tensor(self.text_to_indices(sample["label_text"]), dtype=torch.long)
        return {"image": img, "label": label_encoded}

    def __len__(self):
        return len(self.data)
