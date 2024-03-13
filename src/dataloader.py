from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_loaders(config):
    transform = transforms.Compose([
        transforms.Resize((config.img_h, config.img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std)
    ])

    train_dataset = datasets.ImageFolder(config.train_directory, transform=transform)
    val_dataset = datasets.ImageFolder(config.val_directory, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.n_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.n_workers)
    return train_loader, val_loader
