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

def calculate_pixel_accuracy(generated_img, target_img):
    generated_img = (generated_img.squeeze().detach().cpu().numpy() * 0.5) + 0.5 # Denormalize
    target_img = (target_img.squeeze().detach().cpu().numpy() * 0.5) + 0.5 # Denormalize

    # Thresholding to create binary images (adjust threshold as necessary)
    generated_img = (generated_img > 0.5).astype(int)
    target_img = (target_img > 0.5).astype(int)

    correct_pixels = (generated_img == target_img).sum()
    total_pixels = generated_img.size

    return correct_pixels / total_pixels

def calculate_mean_iou(generated_img, target_img):
    generated_img = (generated_img.squeeze().detach().cpu().numpy() * 0.5) + 0.5  # Denormalize
    target_img = (target_img.squeeze().detach().cpu().numpy() * 0.5) + 0.5  # Denormalize

    # Thresholding to create binary images
    generated_img = (generated_img > 0.5).astype(int)
    target_img = (target_img > 0.5).astype(int)

    intersection = (generated_img & target_img).sum()
    union = (generated_img | target_img).sum()

    return intersection / union if union != 0 else 0
    
if __name__ == "__main__":
    print(device)

    generatorUnet = GeneratorUNet(img_channels=1, z_dim=100).to(device)
    generatorED = GeneratorED(img_channels=1, z_dim=100).to(device)
    # Load the saved state dictionaries
    model = input('Unet model name (without .pth):')
    generatorUnet.load_state_dict(torch.load(f'{model}.pth'))
    model = input('ED model name (without .pth):')
    generatorED.load_state_dict(torch.load(f'{model}.pth'))
    # Set the models to evaluation mode
    generatorUnet.eval()
    generatorED.eval()

    # Define the transformation for the input image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_folder = "test_words/test"
    ans_folder = "test_words/ans"
    test_image_names = [f for f in os.listdir(test_folder) if f.endswith('.png')]
    ans_image_names = [f for f in os.listdir(ans_folder) if f.endswith('.png')]

    total_pixel_accuracy_Unet = 0.0
    total_pixel_accuracy_ED = 0.0
    total_mean_iou_Unet = 0.0
    total_mean_iou_ED = 0.0

    input_images = []
    generated_images_Unet = []
    generated_images_ED = []
    ans_images = []
    for test_img, ans_img in zip(test_image_names, ans_image_names):
        input_image_path = os.path.join(test_folder, test_img)
        ans_image_path = os.path.join(ans_folder, ans_img)


        input_img = load_image(input_image_path, transform=transform)
        ans_image = load_image(ans_image_path, transform=transform)

        input_img = input_img.to(device)
        ans_image = ans_image.to(device)

        z = torch.randn((1, 100)).to(device)  # Random noise vector

        with torch.no_grad():
            generated_img_Unet = generatorUnet(input_img, z)
            generated_img_ED = generatorED(input_img, z)

        input_image = input_img.cpu()
        generated_image_Unet = generated_img_Unet.cpu()
        generated_image_ED = generated_img_ED.cpu()
        ans_image = ans_image.cpu()

        pixel_accuracy_Unet = calculate_pixel_accuracy(generated_image_Unet, ans_image)
        pixel_accuracy_ED = calculate_pixel_accuracy(generated_image_ED, ans_image)

        total_pixel_accuracy_Unet += pixel_accuracy_Unet
        total_pixel_accuracy_ED += pixel_accuracy_ED
        total_mean_iou_Unet += calculate_mean_iou(generated_image_Unet, ans_image)
        total_mean_iou_ED += calculate_mean_iou(generated_image_ED, ans_image)
      
        # Append images for batch display
        input_images.append(input_image)
        generated_images_Unet.append(generated_image_Unet)
        generated_images_ED.append(generated_image_ED)
        ans_images.append(ans_image)

    avg_pixel_accuracy_Unet = total_pixel_accuracy_Unet / len(test_image_names)
    avg_pixel_accuracy_ED = total_pixel_accuracy_ED / len(test_image_names)
    avg_mean_iou_Unet = total_mean_iou_Unet / len(test_image_names)
    avg_mean_iou_ED = total_mean_iou_ED / len(test_image_names)

    print(f"Average Pixel Accuracy for Unet: {avg_pixel_accuracy_Unet:.4f}")
    print(f"Average Pixel Accuracy for ED: {avg_pixel_accuracy_ED:.4f}")
    print(f"Average Mean IoU for Unet: {avg_mean_iou_Unet:.4f}")
    print(f"Average Mean IoU for ED: {avg_mean_iou_ED:.4f}")

    # Convert the list of images to a format suitable for ShowImages
    input_images = torch.stack(input_images)
    generated_images_Unet = torch.stack(generated_images_Unet)
    generated_images_ED = torch.stack(generated_images_ED)
    ans_images = torch.stack(ans_images)
    ShowImages2(input_images, ans_images ,generated_images_ED, generated_images_Unet ,"output_fat")
