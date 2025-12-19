import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root='data/pizza_steak_sushi/train', transform=transform) 
test_data = datasets.ImageFolder(root='data/pizza_steak_sushi/test', transform=transform)


train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=1, 
                              num_workers=1,  
                              shuffle=True) 

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             num_workers=1, 
                             shuffle=False) 

