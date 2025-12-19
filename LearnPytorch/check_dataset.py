import os
from pathlib import Path
directory = 'data/pizza_steak_sushi'
def walk_through_directory(directory):
    """Walk through the directory and print all file paths."""
    for root, dirs, files in os.walk(directory):
        print(f"there are {len(dirs)} directories and {len(files)} files in '{root}'")

walk_through_directory(directory)

# Setup train and testing paths
train_dir = 'data/pizza_steak_sushi/train'
test_dir = 'data/pizza_steak_sushi/test'

train_dir, test_dir

import random 
from PIL import Image

random.seed(42)

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

#get all image paths
image_path_list = list(image_path.glob('*/*/*.jpg'))

random_image_path = random.choice(image_path_list)

random_image_class = random_image_path.parent.stem

img = Image.open(random_image_path)

print(f"Random image path: {random_image_path}")
print(f"Image class: {random_image_class}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")
img.show()

