import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ShowImages(image1, image2, name=None):
    n, m = len(image1), len(image2)
    if n != m:
        return 'bad'
    
    fig, axs = plt.subplots(n, 2)
    for i in range(n):
        axs[i][0].imshow(image1[i][0], cmap='gray')
        axs[i][0].axis('off')
        axs[i][1].imshow(image2[i][0], cmap='gray')
        axs[i][1].axis('off')

    if name is not None:
        fig.savefig(f"6-10-16_35_train_U/{name}.png")
        plt.close()
    else:
        plt.show()