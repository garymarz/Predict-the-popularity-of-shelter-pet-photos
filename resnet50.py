import pandas as pd
import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

f= r"d:\\"
os.chdir(f)
#df = pd.read_csv('PetFinder\\train.csv')
df = pd.read_csv('Kaggle_PetFinder88%\\train.csv')

import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=True, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)[0:8999]
        self.img_dir = img_dir
        self.imgsize =(256 ,256)
        self.transform = transforms.Compose([transforms.Resize(self.imgsize),transforms.ToTensor()])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])+'.jpg'
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, -1]/100 # to 0~1
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
class test_ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=True, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)[9000:]
        self.img_dir = img_dir
        self.imgsize =(256 ,256)
        self.transform = transforms.Compose([transforms.Resize(self.imgsize),transforms.ToTensor()])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])+'.jpg'
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, -1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
		
from torch.utils.data import DataLoader
train_data = ImageDataset(annotations_file='PetFinder\\train.csv',img_dir="PetFinder\\train\\")
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

test_data = test_ImageDataset(annotations_file='PetFinder\\train.csv',img_dir="PetFinder\\train\\")
test_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

from torch import nn
import torchvision.models as models

model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
               nn.Linear(512, 1000),
               nn.ReLU(inplace=True),
               nn.Linear(1000, 125),
                nn.Linear(125, 1))

from torchvision import models
import torch.nn as nn
from torch.optim import Adam

optimizer = Adam(model.parameters(), lr=0.0003)

import wandb
wandb.login()
wandb.init(project='PetFinder_my_Pawpularity Contest')
config = wandb.config
config.learn_rate = 0.0003
config.Model = 'resnet18'
config.Opt = 'Adam'
config.lossfun = 'RMSE'

 training model
import torch.nn.functional as F
minloss = 100
train_losses = []
# empty list to store validation losses
val_losses = []
epochs = 400
wandb.watch(model)
for epoch in range(epochs+1):
    model.train()
    tr_loss = 0
    total =0
    total_train = 0
    # getting the training set
    x_train, y_train = next(iter(train_dataloader))
    # getting the validation set
    x_val, y_val = next(iter(test_dataloader))

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    output_train = model(x_train)
    output_val = model(x_val)
    
    
    # computing the training and validation loss
    criterion = nn.MSELoss()
    output_train = output_train.to(torch.float32)
    y_train = y_train.to(torch.float32)
    loss_train = torch.sqrt(criterion(output_train, y_train))
    loss_val = torch.sqrt(criterion(output_val, y_val))

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    train_loss = loss_train.item()
    wandb.log({'train_loss':loss_train,'val_loss':loss_val})
    
    if epoch%40 == 0:
        # printing the validation loss
        print('Epoch : ',epoch)
        print('train_loss',loss_train.item(),'val_loss :', loss_val.item())
    if minloss > loss_val.item():
        minloss = loss_val.item()
        torch.save(model.state_dict(), 'PetFinder\pet_weight.pt')
        
import torchvision.datasets as dset

test_dataset = dset.ImageFolder(root='PetFinder\\test',
                               transform=transforms.Compose([
                               transforms.Resize(256),
                               transforms.ToTensor()]))
test = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False)

model.eval()
a = []

with torch.no_grad():
    model.eval()
    for i,_ in test:
        y = model(i).numpy()[0][0]
        a.append(y*100)
        #print(y)
d = pd.read_csv('PetFinder//sample_submission.csv')
d['Pawpularity'] = a
d.to_csv("sample_submission.csv")
