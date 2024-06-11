import os
import torch
import torch.nn as nn
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ShowImages(image1, image2,image3, name=None):
    n, m = len(image1), len(image2)
    if n != m:
        return 'bad'
    
    fig, axs = plt.subplots(n, 3)
    for i in range(n):
        axs[i][0].imshow(image1[i][0], cmap='gray')
        axs[i][0].axis('off')
        axs[i][1].imshow(image2[i][0], cmap='gray')
        axs[i][1].axis('off')
        axs[i][2].imshow(image3[i][0], cmap='gray')
        axs[i][2].axis('off')

    if name is not None:
        fig.savefig(f"{name}.png")
        plt.close()
    else:
        plt.show()

def ShowImages2(image1, image2, image3,image4, name=None):
    n = len(image1)
    if n > 5:
        indices = random.sample(range(n), 5)
    else:
        indices = range(n)
    fig, axs = plt.subplots(len(indices), 4)
    for i, idx in enumerate(indices):
        axs[i][0].imshow(image1[idx][0][0], cmap='gray')
        axs[i][0].axis('off')
        axs[i][1].imshow(image2[idx][0][0], cmap='gray')
        axs[i][1].axis('off')
        axs[i][2].imshow(image3[idx][0][0], cmap='gray')
        axs[i][2].axis('off')
        axs[i][3].imshow(image4[idx][0][0], cmap='gray')
        axs[i][3].axis('off')

    fig.savefig(f"test_output/{name}.png")
    plt.close()
