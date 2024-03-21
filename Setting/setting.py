import torch
from torch import nn
from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2
from alphabets import ALPHABETS
from argparse import Namespace

@dataclass
class CoreSettings:
    image_height: int = 32
    image_width: int = 100
    model_save_name: str = "latest_plate_model"
    n_classes: int = 35
    mean_val: list = (0.485,)
    std_val: list = (0.19,)
    alphabet_key: str = "FA_LPR"
    train_path: str = "/home/user/datasets/train"
    val_path: str = "/home/user/datasets/val"
    save_dir: str = "results"

    def refresh_class_count(self):
        self.n_classes = len(self.alphabets) + 1

@dataclass
class TransformSettings(CoreSettings):
    train_aug: A.Compose = A.Compose([
        A.Rotate(limit=12, p=0.25),
        A.RandomScale(scale_limit=0.25),
        A.Resize(height=CoreSettings.image_height, width=CoreSettings.image_width),
        A.Normalize(mean=CoreSettings.mean_val, std=CoreSettings.std_val, max_pixel_value=255.0),
        A.ToGray(always_apply=True),
        ToTensorV2()
    ])
    val_aug: A.Compose = A.Compose([
        A.Resize(height=CoreSettings.image_height, width=CoreSettings.image_width),
        A.Normalize(mean=CoreSettings.mean_val, std=CoreSettings.std_val, max_pixel_value=255.0),
        A.ToGray(always_apply=True),
        ToTensorV2()
    ])

    def refresh_transforms(self):
        self.train_aug = A.Compose([
            A.Rotate(limit=12, p=0.25),
            A.RandomScale(scale_limit=0.25),
            A.Resize(height=self.image_height, width=self.image_width),
            A.Normalize(mean=self.mean_val, std=self.std_val, max_pixel_value=255.0),
            A.ToGray(always_apply=True),
            ToTensorV2()
        ])
        self.val_aug = A.Compose([
            A.Resize(height=self.image_height, width=self.image_width),
            A.Normalize(mean=self.mean_val, std=self.std_val, max_pixel_value=255.0),
            A.ToGray(always_apply=True),
            ToTensorV2()
        ])

@dataclass
class FullConfig(TransformSettings):
    lstm_hidden: int = 256
    lstm_input_size: int = 64
    input_channels: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 0.0005
    lr_patience_epochs: int = 10
    min_lr_val: float = 5e-6
    lr_decay_factor: float = 0.1
    batch_sz: int = 128
    total_epochs: int = 200
    num_workers: int = 8
    alphabets: list = ALPHABETS[CoreSettings.alphabet_key]
    char_to_idx: dict = None
    idx_to_char: dict = None
    early_stop_patience: int = 30

    def apply_args(self, args):
        if isinstance(args, Namespace):
            vars_dict = vars(args)
        elif isinstance(args, dict):
            vars_dict = args
        else:
            raise ValueError("Args must be Namespace or dict.")
        for k, v in vars_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
            elif k == "visualize":
                print(" Skipping visualize argument!")
            else:
                raise ValueError(f"Key {k} not defined in FullConfig")
        self.update_config()

    def update_config(self):
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.alphabets)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.refresh_class_count()
        self.refresh_transforms()

    def as_dict(self):
        cfg = {}
        for key in dir(self):
            val = getattr(self, key)
            if key.startswith("__") or callable(val):
                continue
            cfg[key] = val
        return cfg

    def __repr__(self):
        return f"{self.__class__.__name__}: " + ", ".join(f"{k}={v}" for k, v in self.as_dict().items())
