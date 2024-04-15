import torch
import torchvision
import matplotlib.pyplot as plt

def display_batch_samples(data_loader, mean_vals, std_vals, max_samples=8, samples_per_row=4):
    batch_imgs, batch_labels = next(iter(data_loader))
    batch_imgs = batch_imgs[:max_samples]

    mean_tensor = torch.tensor(mean_vals).view(-1, 1, 1)
    std_tensor = torch.tensor(std_vals).view(-1, 1, 1)
    batch_imgs = batch_imgs * std_tensor + mean_tensor

    img_grid = torchvision.utils.make_grid(batch_imgs, nrow=samples_per_row)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
