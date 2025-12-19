import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import requests
import zipfile
from pathlib import Path

data_path = Path('data')
image_path = data_path / 'pizza_steak_sushi'

if image_path.is_dir():
    print(f"{image_path} directory already exists, skipping download.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / 'pizza_steak_sushi.zip', 'wb') as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza_steak_sushi.zip...")
        f.write(request.content)
    
    #unzip the file
    with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip', 'r') as zip_ref:
        print("Unzipping pizza_steak_sushi.zip...")
        zip_ref.extractall(image_path)
    print("Download and extraction complete.")