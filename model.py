import os
import torch
from torchvision import transforms
from PIL import Image
from train_U import GeneratorUNet
from train_GAN import GeneratorED
from utils import ShowImages2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    if transform is not None:
        image = transform(image).unsqueeze(0)    # Add batch dimension
    return image

if __name__ == "__main__":
    print(device)

    generator = GeneratorUNet(img_channels=1, z_dim=100)

    # Load the saved state dictionaries
    model = input('model name (without .pth): ')
    generator.load_state_dict(torch.load(f'{model}.pth'))

    # Set the models to evaluation mode
    generator.eval()

    # Define the transformation for the input image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    input_image_path = '.png'
    input_img = load_image(input_image_path).to(device)
    z = torch.randn((1, 100)).to(device)  # Random noise vector
    with torch.no_grad():
        generated_img = generator(input_img, z)

    input_image = input_img.cpu()
    generated_image = generated_img.cpu()
    ShowImages2(input_image, generated_image)