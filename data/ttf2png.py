import os
from PIL import Image, ImageDraw, ImageFont

# Load the font
font_path = "ttf/.ttf"
font_size = 50
try:
    font = ImageFont.truetype(font_path, font_size)  # default encoding: 'unic'
except IOError:
    print(f"Could not load font at {font_path}.")
    exit()

# Create a directory to save the images
output_dir = "font/{font_name}"
os.makedirs(output_dir, exist_ok=True)

# Open and read the words from words.txt
with open("words.txt", "r") as file:
    words = file.readlines()

# Set image size
image_width = 64
image_height = 64
for i, word in enumerate(words):
    if (i >= 2000):
        break
    word = word.strip()  # Remove any leading/trailing whitespace

    # Create a new image in grayscale mode ("L") with a white background
    image = Image.new("L", (image_width, image_height), 255)
    draw = ImageDraw.Draw(image)

    # Draw the word on the image in black
    draw.text((6, 0), word, font=font, fill=0)

    # Save the image
    image_path = os.path.join(output_dir, f"word{i + 1}.png")
    image.save(image_path)

print("All images have been saved.")