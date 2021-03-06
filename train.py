import numpy as np
import torch
import yaml
from torch.utils.data import random_split, DataLoader
from torch.nn import BCELoss
from torch.optim import Adam, lr_scheduler
from torch import device, save
from model import BeatTrackingNet
from data import BallroomDataset
from torch.utils.data import DataLoader
import pdb

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# cuda
if torch.cuda.is_available():
    GPU = True
else:
    GPU = False

is_load = False

# Training parameters
is_train = True
cross_validation = False

num_epoch = config['num_epoch']
batch_size = config['batch_size']
optimizer = config['optimizer']
learning_rate = config['learning_rate']
k_fold = config['k_fold']

# load dataset
dataset = BallroomDataset()
# split dataset
# if not cross_validation:
[train_dataset, valid_dataset] = random_split(dataset, [9, 1],
                                              generator=torch.Generator().manual_seed(42))

valid_loader = DataLoader(valid_dataset, batch_size = batch_size)

model = BeatTrackingNet()
parameters = model.parameters()

if GPU:
    model = model.cuda()

if optimizer == 'Adam':
    optimizer = Adam(parameters, lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2)
criterion = BCELoss()

params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Total parameters: ')
# train and valid
for i in range(1, num_epoch + 1):
    # training
    print(f"Epoch {i}: Training Start.")
    model.train()
    running_loss = 0.0
    batch_loss = list()
    batch_step = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    for input, label in train_loader:
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch_loss.append(loss.item())
        batch_step += 1
        if batch_step % 20 == 1:
            print(f'Epoch {i}, Step {batch_step}; '
                  f'Average Train Loss {running_loss / (batch_step * batch_size)}.')

    # validation
    model.eval()
    print(f"Epoch {i}: Validation Start.")
    train_loader = DataLoader(train_dataset, batch_size=len(valid_dataset))
    with torch.no_grad():
        for input, label in valid_loader:
            output = model(input)
            loss = criterion(output, label)
            print(f'Average Valid Loss {loss.item() / len(valid_dataset)}.')
            break

    # save model
    if i % 2 == 0:
        torch.save(model.cpu().state_dict(), f"{config['model_folder']}_Epoch{i}.pt")

# evaluate

# test
