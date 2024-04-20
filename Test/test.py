import torch
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset_module import OCRDataset, compute_mean_std, extract_text_from_filename, validate_text, prepare_dataset
from config_module import Config
from model_module import LitCRNN
from dataloader_module import get_loaders
from logger_module import initialize_logger
from utils_module import mkdir_incremental, visualize_data_loader


def test_dataset():
    print("=== Testing OCRDataset ===")
    sample_alphabet = "ABCD0123"
    dummy_dir = Path("./dummy_dataset")
    dummy_dir.mkdir(exist_ok=True)
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (32, 100), dtype=np.uint8))
        img.save(dummy_dir / f"A{i}.png")
    dataset = OCRDataset(str(dummy_dir), sample_alphabet)
    print("Length:", len(dataset))
    img, label = dataset[0]
    print("Image shape:", img.size())
    print("Label:", label)



def test_transforms():
    print("\n=== Testing Transforms ===")
    img_h, img_w = 32, 100
    transforms = A.Compose([
        A.Resize(img_h, img_w),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    img = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)
    transformed = transforms(image=img)['image']
    print("Transformed image shape:", transformed.shape)


def test_config():
    print("\n=== Testing Config ===")
    cfg = Config()
    print(cfg.__dict__)
    cfg.update_config_param({"batch_size": 64, "img_w": 120})
    print("Updated batch size:", cfg.batch_size, "Updated img_w:", cfg.img_w)


def test_model_forward():
    print("\n=== Testing LitCRNN forward ===")
    cfg = Config()
    model = LitCRNN(cfg.img_h, cfg.n_channels, cfg.n_classes, cfg.n_hidden, cfg.lstm_input,
                    cfg.lr, cfg.lr_reduce_factor, cfg.lr_patience, cfg.min_lr)
    dummy_input = torch.randn(2, cfg.n_channels, cfg.img_h, cfg.img_w)
    output = model(dummy_input)
    print("Output shape:", output.shape)

def test_logger():
    print("\n=== Testing Logger ===")
    log = initialize_logger("TestLogger")
    log.info("This is a test log message.")
    
def test_mkdir_incremental():
    print("\n=== Testing mkdir_incremental ===")
    path = Path("./test_output")
    new_path = mkdir_incremental(path)
    print("New directory created at:", new_path)