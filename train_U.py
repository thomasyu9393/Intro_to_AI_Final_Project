import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import FontDataset
from utils import ShowImages
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

        self.down1 = UNetDown(img_channels, 64, normalize=False)  # (1, 64, 32, 32)
        self.down2 = UNetDown(64, 128)                            # (1, 128, 16, 16)
        self.down3 = UNetDown(128, 128)                           # (1, 128, 8, 8)
        self.down4 = UNetDown(128, 128)                           # (1, 128, 4, 4)
        self.down5 = UNetDown(128, 128)                           # (1, 128, 2, 2)
        self.down6 = UNetDown(128, 128)                           # (1, 128, 1, 1)

        self.fc = nn.Linear(128 * 1 * 1 + z_dim, 128 * 1 * 1)
        self.fc_reshape = nn.Sequential(
            nn.BatchNorm1d(128 * 1 * 1),
            nn.ReLU(True),
        )

        self.up1 = UNetUp(128, 128)
        self.up2 = UNetUp(256, 128)
        self.up3 = UNetUp(256, 128)
        self.up4 = UNetUp(256, 128)
        self.up5 = UNetUp(256, 64)
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(128, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, img, z):
        # Encoder
        d1 = self.down1(img)  # (1, 64, 32, 32)
        d2 = self.down2(d1)   # (1, 128, 16, 16)
        d3 = self.down3(d2)   # (1, 128, 8, 8)
        d4 = self.down4(d3)   # (1, 128, 4, 4)
        d5 = self.down5(d4)   # (1, 128, 2, 2)
        d6 = self.down6(d5)   # (1, 128, 1, 1)

        # Flatten and combine with z
        d6_flat = d6.view(d6.size(0), -1)          # (1, 128 * 1 * 1) = (1, 128)
        z = z.view(z.size(0), -1)                  # (1, z_dim)
        fc_input = torch.cat((d6_flat, z), dim=1)  # (1, 128 + z_dim) = (1, 228)

        # Fully connected bottleneck
        fc_output = self.fc(fc_input)                             # (1, 128 * 1 * 1) = (1, 128)
        fc_output = self.fc_reshape(fc_output)                    # (1, 128)
        fc_output = fc_output.view(fc_output.size(0), 128, 1, 1)  # (1, 128, 1, 1)

        # Decoder
        u1 = self.up1(fc_output, d5)  # (1, 256, 2, 2)
        u2 = self.up2(u1, d4)         # (1, 128 + 128 = 256, 4, 4)
        u3 = self.up3(u2, d3)         # (1, 128 + 128 = 256, 8, 8)
        u4 = self.up4(u3, d2)         # (1, 128 + 128 = 256, 16, 16)
        u5 = self.up5(u4, d1)         # (1, 64 + 64 = 128, 32, 32)
        u6 = self.up7(u5)             # (1, 1, 64, 64)

        return u6

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
    print(device)

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure the images are grayscale
    #    transforms.Resize((64, 64)),  # Resize the images to 64x64
        transforms.ToTensor(),  # Convert the images to tensors
        transforms.Normalize([0.5], [0.5])  # Normalize the images
    ])

    # Define the directory containing the images
    root_dir = 'data/ChineseChar/'
    font_dirs = ['LXGWWenKaiTC-Regular', 'HanWangShinSu-Medium']
    dataset = FontDataset(root_dir=root_dir, font_dirs=font_dirs, transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the generator and discriminator
    img_channels = 1
    z_dim = 100
    generator = GeneratorUNet(img_channels, z_dim).to(device)
    discriminator = Discriminator(img_channels).to(device)

    # Define the loss function and optimizers
    BCE_Loss = nn.BCELoss().to(device)
    L1_Loss = nn.L1Loss().to(device)
    LAMBDA = 100

    lr = 0.0002
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    print('Start!!!')

    losses = []

    # Training loop
    num_epochs = 2000
    for epoch in range(num_epochs):
        for i, (input_img, target_img) in enumerate(dataloader):
            batch_size = input_img.size(0)
            real = torch.ones(batch_size, 1)
            fake = torch.zeros(batch_size, 1)
            
            # Configure input
            base_imgs = input_img.to(device)
            real_imgs = target_img.to(device)
            real = real.to(device)
            fake = fake.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real images
            outputs_real = discriminator(real_imgs)
            d_loss_real = BCE_Loss(outputs_real, real)

            # Fake images
            z = torch.randn(batch_size, z_dim).to(device)  # Random noise
            fake_imgs = generator(base_imgs, z)
            outputs_fake = discriminator(fake_imgs.detach())
            d_loss_fake = BCE_Loss(outputs_fake, fake)

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
            
            gan_loss = BCE_Loss(outputs_fake, real)
            l1_loss = L1_Loss(fake_imgs, real_imgs)
            total_gen_loss = gan_loss + (LAMBDA * l1_loss)

            total_gen_loss.backward()
            optimizer_G.step()

        # Print the progress
        print(f"[Epoch {epoch+1}/{num_epochs}] [D loss: {d_loss.item()}] [G loss: {total_gen_loss.item()}]")

        base_imgs_show = base_imgs.cpu()
        fake_imgs_show = fake_imgs.data.cpu()
        real_imgs_show = real_imgs.data.cpu()
        ShowImages(base_imgs_show[:5], fake_imgs_show[:5], real_imgs_show[:5], name=None)


        losses.append((d_loss.item(), total_gen_loss.item()))
        temp_losses = list(zip(*losses))
        plt.figure()
        plt.plot(range(0, epoch + 1), temp_losses[0], label='Discriminator Loss')
        plt.plot(range(0, epoch + 1), temp_losses[1], label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        #plt.savefig('6-10-16_33_train_U/training_loss.png')
        plt.close()

        if epoch % 400 == 0:
            torch.save(generator.state_dict(), f'UNet_{epoch}.pth')
