import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import FontDataset
from utils import ShowImages
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Encoder(nn.Module):
    def __init__(self, img_channels):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),  # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # (512, 2, 2)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # (512, 1, 1)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, img_channels):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # (512, 2, 2)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),  # (img_channels, 64, 64)
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
    
class GeneratorED(nn.Module):
    def __init__(self, img_channels=1, z_dim=100):
        super(GeneratorED, self).__init__()
        self.encoder = Encoder(img_channels)
        self.fc1 = nn.Linear(512 + z_dim, 512 * 1 * 1)
        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(512 * 1 * 1),
            nn.ReLU(True),
        )
        self.decoder = Decoder(img_channels)

    def forward(self, img, z):
        # Encoder
        enc_output = self.encoder(img)  # (N, 512, 1, 1)
        enc_output = enc_output.view(enc_output.size(0), -1)  # Flatten: (N, 512)
        
        # Concatenate with random noise vector z
        z = z.view(z.size(0), -1)  # (N, z_dim)
        combined = torch.cat((enc_output, z), dim=1)  # (N, 512 + z_dim)
        
        # Fully connected layers
        fc_output = self.fc1(combined)  # (N, 512 * 1 * 1)
        fc_output = self.fc2(fc_output)  # (N, 512 * 1 * 1)
        fc_output = fc_output.view(fc_output.size(0), 512, 1, 1)  # Reshape: (N, 512, 1, 1)
        
        # Decoder
        dec_output = self.decoder(fc_output)  # (N, img_channels, 64, 64)
        
        return dec_output

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
    root_dir = 'data/font/'
    font_dirs = ['LXGWWenKaiTC-Regular', 'HanWangShinSu-Medium']
    dataset = FontDataset(root_dir=root_dir, font_dirs=font_dirs, transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the generator and discriminator
    img_channels = 1
    z_dim = 100
    generator = GeneratorED(img_channels, z_dim).to(device)
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
    num_epochs = 100
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

