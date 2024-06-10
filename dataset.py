import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import ShowImages

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label

class FontDataset(Dataset):
    def __init__(self, root_dir, font_dirs, transform=None):
        self.root_dir = root_dir
        self.font_dirs = font_dirs
        self.transform = transform

        # Assume all font directories have the same number of images with matching names
        self.word_names = sorted(os.listdir(os.path.join(root_dir, font_dirs[0])))

    def __len__(self):
        return len(self.word_names)

    def __getitem__(self, idx):
        word_name = self.word_names[idx]
        images = []

        for font_dir in self.font_dirs:
            img_name = os.path.join(self.root_dir, font_dir, word_name)
            image = Image.open(img_name).convert("L")  # Convert image to grayscale
            if self.transform:
                image = self.transform(image)
            images.append(image)

        # images[0] is input, images[1] is target
        return images[0], images[1]
    
if __name__ == '__main__':

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create the dataset
    root_dir = 'data/ChineseChar/'
    font_dirs = ['LXGWWenKaiTC-Regular', 'HanWangShinSu-Medium']
    dataset = FontDataset(root_dir=root_dir, font_dirs=font_dirs, transform=transform)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    print('Load Finished.')

    for i, (input_img, target_img) in enumerate(dataloader):
        ShowImages(input_img, target_img)

        if i == 0:
            break
        
