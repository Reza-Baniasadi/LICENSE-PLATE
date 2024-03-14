import torch
import torchvision
import matplotlib.pyplot as plt


def visualize_data_loader(loader, mean, std):
    imgs, labels = next(iter(loader))
    imgs = imgs[:8]
    imgs = imgs * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)

    grid = torchvision.utils.make_grid(imgs, nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
