import os
import re
from os.path import join
from argparse import ArgumentParser

import cv2
import albumentations as A
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from deep_utils import split_extension, log_print


def check_text_validity(sentence: str, valid_chars: str):
    wrong_chars = set(sentence) - set(valid_chars)
    return len(wrong_chars) == 0, (None if not wrong_chars else list(wrong_chars)[0])


def filename_to_text(fname: str) -> str:
    return os.path.splitext(fname)[0]


def build_dataset_paths(root_dir, vocab, char_to_index, logger=None):
    img_files, encoded_labels, lengths = [], [], []
    skip_count = 0

    for f in os.listdir(root_dir):
        fpath = join(root_dir, f)
        try:
            if split_extension(f)[-1].lower() in [".jpg", ".jpeg", ".png"]:
                text = filename_to_text(f)
                is_valid, bad = check_text_validity(text, vocab)
                if is_valid:
                    enc = [char_to_index[c] for c in text]
                    img_files.append(fpath)
                    encoded_labels.append(enc)
                    lengths.append(len(enc))
                else:
                    log_print(logger, f"[Skipped] {fpath} -> invalid char: {bad}")
                    skip_count += 1
            else:
                log_print(logger, f"[Skipped] {fpath} unsupported format")
                skip_count += 1
        except Exception as e:
            log_print(logger, f"[Error] {fpath}: {e}")
            skip_count += 1

    log_print(logger, f"Collected {len(img_files)} samples, skipped {skip_count}")
    return img_files, encoded_labels, lengths


class TextImageDataset(Dataset):
    def __init__(self, root_dir, alphabet, transform=None, logger=None):
        self.char_to_idx = {c: i + 1 for i, c in enumerate(alphabet)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        self.paths, self.targets, self.target_sizes = build_dataset_paths(
            root_dir, alphabet, self.char_to_idx, logger=logger
        )

        self.transform = transform
        self.num_classes = len(self.idx_to_char) + 1

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        img = np.array(Image.open(img_path))[..., :3]

        if isinstance(self.transform, A.Compose):
            img = self.transform(image=img)["image"][0:1, ...].unsqueeze(0)
        else:
            img = self.transform(Image.fromarray(img)).unsqueeze(0)

        label = torch.LongTensor(self.targets[index]).unsqueeze(0)
        length = torch.LongTensor([self.target_sizes[index]]).unsqueeze(0)
        return img, label, length

    @staticmethod
    def merge_batch(batch):
        imgs, lbls, lens = zip(*batch)
        imgs = torch.cat(imgs, dim=0)
        lbls = [l.squeeze(0) for l in lbls]
        lbls = nn.utils.rnn.pad_sequence(lbls, padding_value=-100).T
        lens = torch.cat(lens, dim=0)
        return imgs, lbls, lens


def calc_dataset_statistics(data_dir, alphabet, batch_size, img_h, img_w):
    tx = T.Compose([
        T.Grayscale(),
        T.Resize((img_h, img_w)),
        T.ToTensor()
    ])

    ds = TextImageDataset(root_dir=data_dir, alphabet=alphabet, transform=tx)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=ds.merge_batch)

    total_mean, total_std = torch.zeros(1), torch.zeros(1)
    n = len(ds)

    for imgs, _, _ in tqdm(loader, desc="Computing stats"):
        total_mean += imgs.mean(dim=(0, 2, 3)).sum()
        total_std += imgs.std(dim=(0, 2, 3)).sum()

    mean_val = (total_mean / n).item()
    std_val = (total_std / n).item()
    return [round(mean_val, 4)], [round(std_val, 4)]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", help="dataset directory path")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--alphabet", default="ابپتشثجدزسصطعفقکگلمنوهی+۰۱۲۳۴۵۶۷۸۹")
    parser.add_argument("--img_h", type=int, default=32)
    parser.add_argument("--img_w", type=int, default=128)

    args = parser.parse_args()
    mean, std = calc_dataset_statistics(
        args.data_dir, args.alphabet, args.batch_size, args.img_h, args.img_w
    )
    print("Mean:", mean, "Std:", std)
