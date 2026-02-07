from math import gamma
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda')

BATCHES = 8
EPOCHS = 20
RATE = 0.001

data_transforms = {
        'train': transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.255]
                    )
                ]
            ),
        'val': transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.255]
                    )
                ]
            )
        }

wafers = 'dataset'
images = {x: datasets.ImageFolder(os.path.join(wafers, x), data_transforms[x]) for x in ['train', 'val']}
loaders = {x: DataLoader(images[x], batch_size=BATCHES, shuffle=True) for x in ['train', 'val']}

class_names = images['train'].classes
dataset_size = {x: len(images[x]) for x in ['train', 'val']}

model = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=RATE, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, epochs=25):
    best = 0.0 

    for epoch in range(epochs):
        print(f'Epoch: {epoch}/{epochs}\n{"-" * 10}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            rloss, rcorrects = 0.0, 0

            for inputs, labels in loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                rloss += loss.item() * inputs.size(0)
                rcorrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = rloss / dataset_size[phase]
            epoch_acc = float(rcorrects) / dataset_size[phase]

            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

            if phase == 'val' and epoch_acc > best:
                best = epoch_acc
                torch.save(model.state_dict(), 'wafer_model.pth')

    print(f'Best val acc: {best}')
    return model

model = train_model(model, criterion, optimizer, exp_lr_scheduler)
