import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def build_data_pipelines(cfg):
    pipeline = transforms.Compose([
        transforms.Resize((cfg.img_height, cfg.img_width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.channel_mean, std=cfg.channel_std)
    ])

    train_data = datasets.ImageFolder(cfg.train_path, transform=pipeline)
    validation_data = datasets.ImageFolder(cfg.val_path, transform=pipeline)

    train_iterator = DataLoader(
        train_data,
        batch_size=cfg.batch_sz,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    val_iterator = DataLoader(
        validation_data,
        batch_size=cfg.batch_sz,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    return train_iterator, val_iterator
