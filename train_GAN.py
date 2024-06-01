import torch
import torch.nn as nn
from torch.autograd import Variable
from v3 import Encoder, Decoder, Autoencoder
import load_data
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is (1) x 50 x 50
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # state size. (64) x 25 x 25
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # state size. (128) x 13 x 13
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # state size. (256) x 7 x 7
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            # state size. (1) x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.main(input)
        x = x.view(-1)
        return x

"""
class Generator(nn.Module):
    def __init__(self, z_dim):

        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. (512) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (256) x 8 x 8
            nn.ConvTranspose2d( 256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (64) x 11 x 11
            nn.ConvTranspose2d( 128, 64, 4, 2, 0, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (32) x 25 x 25
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),   
            # state size. (64) x 50 x 50    
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
"""
class Generator(nn.Module):
    def __init__(self, noise_dim, latent_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 50 * 50),  # Output size for 50x50 grayscale image
            nn.Sigmoid()
        )

    def forward(self, noise, latent_vector):
        x = torch.cat((noise, latent_vector), dim=1)
        x = self.fc(x)
        x = x.view(-1, 1, 50, 50)  # Reshape the output to image dimensions
        return x


# Initialize models
latent_dim = 128
noise_dim = 100
generator = Generator(latent_dim, noise_dim).to(device)
discriminator = Discriminator().to(device)
criterion = nn.MSELoss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

def save_images(epoch, image1, image2):
    fig, axs = plt.subplots(5, 2)
    for i in range(5):
        axs[i, 0].imshow(image1[i][0][0], cmap='gray')
        axs[i, 0].axis('off')
        axs[i, 1].imshow(image2[i][0][0], cmap='gray')
        axs[i, 1].axis('off')
    
    fig.savefig(f"seepa/GAN0_{epoch}.png")
    plt.close()

# Main
if __name__ == '__main__':
    print(device, 'train_GAN.py')

    autoencoder = Autoencoder(latent_dim=128)
    autoencoder.load_state_dict(torch.load('autoencoder_v3.pth'))
    autoencoder.eval()

    def get_latent(encoder, images):
        with torch.no_grad():
            latents = encoder(images)
        return latents

    batch_size = 32

    data = load_data.load_images_in_tensor(dataset_dir='Traditional_Chinese_Data')
    train_data, test_data = load_data.split_dataset(data, train_size=0.2)
    train_batches = load_data.create_batches(train_data, batch_size=32)
    print('Start!!!!!')

    def jizz(li):
        a = []
        for (i, j) in li:
            a.append(i)
        return torch.stack(a)

    num_epochs = 100
    for epoch in range(num_epochs):
        wtf = None
        for batch in train_batches:
            images = []
            for (image, label) in batch:
                images.append(image)
            images = torch.stack(images).to(device)

            generator.eval()
            discriminator.train()

            # Init gradient
            optimizer_D.zero_grad()
            # print(len(images))
            real_validity = discriminator(images).to(device)
            # print(real_validity)
            real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))

            # Building z
            z = Variable(torch.randn(batch_size, noise_dim)).to(device)
            fake_images = generator(z, get_latent(autoencoder.encoder, images).to(device))
            fake_validity = discriminator(fake_images).to(device)
            fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))

            d_loss = real_loss + fake_loss
            d_loss.backward()

            optimizer_D.step()
            d_loss_data = d_loss.data

            if epoch and epoch % 10 == 0:
                generator.train()
                discriminator.eval()

                optimizer_G.zero_grad()

                # Building z2
                z2 = Variable(torch.randn(batch_size, noise_dim)).to(device)
                g_image = generator(z2, get_latent(autoencoder.encoder, images).to(device))
                g_validity = discriminator(g_image)
                wtf = g_validity.detach().cpu()

                g_loss = criterion(g_validity, Variable(torch.ones(batch_size)).to(device))

                g_loss.backward()
                optimizer_G.step()
                g_loss_data = g_loss.data

        if epoch==0 or epoch % 10:
            print(f"Epoch [{epoch + 1}/{num_epochs}] | D Loss: {d_loss_data}")
        else:
            print(f"Epoch [{epoch + 1}/{num_epochs}] | D Loss: {d_loss_data} | G Loss: {g_loss_data}")
            print(wtf)

        # Set generator eval
        generator.eval()
        
        # Building z 
        z = Variable(torch.randn(5, noise_dim)).to(device)
        res_data = jizz(data[:5]).to(device)
        gen_data = generator(z, get_latent(autoencoder.encoder, res_data).to(device)).detach()
        print(res_data.shape)
        print(gen_data.shape)
        gen_data = gen_data.unsqueeze(1).data.cpu()
        res_data = res_data.unsqueeze(1).data.cpu()

        save_images(epoch, res_data, gen_data)

        