import os
import random
from PIL import Image
import torch
from torchvision import transforms
import random
import numpy as np

def image_transform(image, path=0):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5,), std = (0.5,))
    ])
    if path:
        image = Image.open(image)
    return transform(image)

def load_images_in_tensor(dataset_dir):
    data = []
    for word in os.listdir(dataset_dir):
        word_dir = os.path.join(dataset_dir, word)
        if os.path.isdir(word_dir):
            for img_name in os.listdir(word_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(word_dir, img_name)
                    data.append((image_transform(img_path, path=1), word))
    return data

def split_dataset(data, train_size=0.9):
    random.shuffle(data)
    train_size = int(train_size * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def create_batches(data, batch_size):
    random.shuffle(data)
    batches = [data[i:i + batch_size] for i in range(0, len(data) - batch_size + 1, batch_size)]
    return batches

def create_pairs_for_single_word(data, batch_size):
    """
    Create pairs of images for a single word.
    
    Args:
    data (list): A list of data samples.
    batch_size (int): Number of samples per batch.
    
    Returns:
    list: A list of batches with image pairs and labels.
    """
    word_to_images = {}
    for image, label in data:
        if label not in word_to_images:
            word_to_images[label] = []
        word_to_images[label].append(image)
    
    pairs = []
    for _ in range(batch_size):
        # Randomly pick a word (label)
        label = random.choice(list(word_to_images.keys()))
        images = word_to_images[label]
        
        # Randomly pick two images from the chosen word
        img1, img2 = random.sample(images, 2)
        pairs.append((img1, img2, label))
    
    return pairs

def main():
    dataset_dir = 'Traditional_Chinese_Data'
    batch_size = 32
    data = load_images_in_tensor(dataset_dir)
    train_data, test_data = split_dataset(data)
    
    # First way: Shuffle and sample batches of words
    train_batches = create_batches(train_data, batch_size)
    test_batches = create_batches(test_data, batch_size)
    
    # Second way: Create pairs for a single word
    train_pairs = create_pairs_for_single_word(train_data, batch_size)
    test_pairs = create_pairs_for_single_word(test_data, batch_size)
    
    # Print some examples to verify
    print(f"Train batch example: {train_batches[0]}")
    print(f"Test batch example: {test_batches[0]}")
    print(f"Train pairs example: {train_pairs[:2]}")
    print(f"Test pairs example: {test_pairs[:2]}")

if __name__ == "__main__":
    main()
