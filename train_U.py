import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), dim=1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, img_channels=1, z_dim=100):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(img_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 + z_dim, 512, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(128, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, img, z):
        # Encoder
        d1 = self.down1(img)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        # Add z to bottleneck
        z = z.view(z.size(0), z.size(1), 1, 1)
        z = z.expand(z.size(0), z.size(1), d7.size(2), d7.size(3))
        bottleneck_input = torch.cat((d7, z), dim=1)
        bottleneck_output = self.bottleneck(bottleneck_input)

        # Decoder
        u1 = self.up1(bottleneck_output, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6)

        return u7

class Discriminator(nn.Module):
    def __init__(self, img_channels=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # 1x1
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(img.size(0), -1)



if __name__ == '__main__':
    # Define the directory containing the images
    data_dir = 'LXGWWenKaiTC-Regular'

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure the images are grayscale
    #    transforms.Resize((64, 64)),  # Resize the images to 64x64
        transforms.ToTensor(),  # Convert the images to tensors
        transforms.Normalize([0.5], [0.5])  # Normalize the images
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    # Initialize the generator and discriminator
    img_channels = 1
    z_dim = 100
    generator = GeneratorUNet(img_channels, z_dim).to(device)
    discriminator = Discriminator(img_channels).to(device)

    # Define the loss function and optimizers
    criterion = nn.BCELoss().to(device)
    lr = 0.0002
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999)).to(device)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999)).to(device)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real = torch.ones(batch_size, 1)
            fake = torch.zeros(batch_size, 1)
            
            # Configure input
            real_imgs = imgs.to(device)  # Move images to GPU
            real = real.to(device)
            fake = fake.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real images
            outputs_real = discriminator(real_imgs)
            d_loss_real = criterion(outputs_real, real)

            # Fake images
            z = torch.randn(batch_size, z_dim).to(device)  # Random noise
            fake_imgs = generator(real_imgs, z)
            outputs_fake = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(outputs_fake, fake)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake images and calculate loss
            outputs_fake = discriminator(fake_imgs)
            g_loss = criterion(outputs_fake, real)

            g_loss.backward()
            optimizer_G.step()

            # Print the progress
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")