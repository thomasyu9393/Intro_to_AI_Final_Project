import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt
import numpy as np
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # [1, 50, 50]
            nn.Conv2d(1, 32, kernel_size=3, stride=1).to(device),
            # [32, 48, 48]
            nn.ReLU().to(device),
            nn.MaxPool2d(2, 2).to(device),
            # [32, 24, 24]
            nn.Conv2d(32, 64, kernel_size=3, stride=1).to(device),
            # [64, 22, 22]
            nn.ReLU().to(device),
            nn.MaxPool2d(2, 2).to(device),
            # [64, 11, 11]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1).to(device),
            # [128, 10, 10]
            nn.ReLU().to(device),
            nn.MaxPool2d(2, 2).to(device),
            # [128, 5, 5]
            nn.Flatten().to(device),
            nn.Linear(128 * 5 * 5, latent_dim).to(device),
            nn.ReLU().to(device)
        )
    
    def forward(self, x):
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 5 * 5).to(device),
            nn.ReLU().to(device),
            nn.Unflatten(1, (128, 5, 5)).to(device),
            # [128, 5, 5]
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=3, padding=1).to(device),
            # [64, 13, 13]
            nn.ReLU().to(device),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1).to(device),
            # [32, 25, 25]
            nn.ReLU().to(device),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1).to(device),
            # [1, 50, 50]
            nn.Sigmoid().to(device)
        )
    
    def forward(self, x):
        return self.decoder(x)
    
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
def main():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((50, 50)),
        transforms.ToTensor()
    ])

    dataset0 = datasets.ImageFolder(root='Traditional_Chinese_Data', transform=transform)
    train_size = int(0.9 * len(dataset0))
    train_dataset0, test_dataset0 = random_split(dataset0, [train_size, len(dataset0) - train_size])

    train_dataset = DataLoader(train_dataset0, batch_size=32, shuffle=True)
    test_dataset = DataLoader(test_dataset0, batch_size=1, shuffle=False)

    latent_dim = 128
    autoencoder = Autoencoder(latent_dim).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0002)

    print(device, 'v3.py')
    losses = []
    def save_images(epoch):
        # Get a batch of test images
        # subdataset0 = Subset(dataset0, random.sample(range(len(dataset0)), 5))
        # subdataset = DataLoader(subdataset0, batch_size=1, shuffle=False)
        fig, axs = plt.subplots(5, 2)
        for i, data in enumerate(test_dataset):
            if i >= 5:
                break
            img, _ = data
            img = img.to(device)
            rec = autoencoder(img)
            img = img.cpu()
            rec = rec.cpu().detach()

            axs[i, 0].imshow(img.numpy().squeeze(), cmap='gray')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(rec.numpy().squeeze(), cmap='gray')
            axs[i, 1].axis('off')
        
        fig.savefig(f"v3/AE0{epoch}.png")
        plt.close()
        
        plt.figure()
        plt.plot(range(0, epoch + 1), losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.savefig('training_loss_v3.png')
        plt.close()
        
        torch.save(autoencoder.state_dict(), 'autoencoder_v3.pth')

    num_epochs = 3000
    for epoch in range(num_epochs):
        for data in train_dataset:
            imgs, _ = data
            imgs = imgs.to(device)
            
            # Forward pass
            outputs = autoencoder(imgs)

            # Calculate loss with MSE
            loss = criterion(outputs, imgs)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        save_images(epoch)

if __name__ == '__main__':
    main()