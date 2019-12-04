"""
@author: Saidul Islam

usage: python train.py flowers --GPU=GPU
"""

import json
import time

from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

import argparse

# command line arguments for the script
parser = argparse.ArgumentParser ()

parser.add_argument('data_dir', help = 'Provide your data directory.', type = str)
parser.add_argument('--arch', help = 'Provide a pretrained network. Currently supported vgg13 and resnet50. Default resnet50.', type = str)
parser.add_argument('--save_dir', help = 'Provide a save directory for your checkpoints', type = str)
parser.add_argument('--learning_rate', help = 'Provide a learning rate. Default value 0.001', type = float)
parser.add_argument('--hidden_units', help = 'Provide the number of hidden units in Classifier. Default value is 500', type = int)
parser.add_argument('--epochs', help = 'Number of epochs. Default value is 3', type = int)
parser.add_argument('--GPU', help = 'Option to use GPU/CPU. Default value is CPU', type = str)

# get command line args for our training
args = parser.parse_args ()

data_dir = 'flowers'
if args.data_dir:
    data_dir = args.data_dir
    
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

if args.arch:
    arch = args.arch
else:
    arch = 'resnet50'
    
if args.save_dir:
    save_dir = args.save_dir
    
if args.learning_rate:
    learning_rate = args.learning_rate
else:
    learning_rate = .001

if args.hidden_units:
    hidden_units = args.hidden_units
else:
    hidden_units = 500
    
if args.epochs:
    epochs = args.epochs
else:
    epochs = 3
    
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'
    

'''
**** Load the data ****
'''

# Define your transforms for the training, validation, and testing sets
# Define transforms
data_transforms = {}

# Define your transforms for the training
data_transforms['train'] = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# Define your transforms for the validation
data_transforms['valid'] = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# Define your transforms for the test
data_transforms['test'] = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# Load the datasets with ImageFolder
image_datasets = {}
image_datasets['train_data'] = datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train'])
image_datasets['valid_data'] = datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['valid'])
image_datasets['test_data'] = datasets.ImageFolder(data_dir + '/test', transform=data_transforms['test'])

# Using the image datasets and the transforms, define the dataloaders
dataloaders = {}
dataloaders['train_data'] = DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True)
dataloaders['valid_data'] = DataLoader(image_datasets['valid_data'], batch_size=32, shuffle=False)
dataloaders['test_data'] = DataLoader(image_datasets['test_data'], batch_size=32, shuffle=False)

print(f"Train data: {len(dataloaders['train_data'].dataset)} images / {len(dataloaders['train_data'])} batches")
print(f"Valid data: {len(dataloaders['valid_data'].dataset)} images / {len(dataloaders['valid_data'])} batches")
print(f"Test  data: {len(dataloaders['test_data'].dataset)} images / {len(dataloaders['test_data'])} batches")


'''
**** Label mapping ****
'''
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


'''
**** Load model ****
'''
def load_model (arch, hidden_units):
    if arch == 'resnet50':
        #setting model based on vgg13
        model = models.resnet50(pretrained=True)
        
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
            
        classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (2048, 1000)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.5)),
                            ('fc2', nn.Linear (1000, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.5)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        model.classifier = classifier
        model.fc = classifier
        
    else: 
        #setting model based on default vgg13
        model = models.vgg13(pretrained = True)
        
        for param in model.parameters():
            param.requires_grad = False
            
        classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        model.classifier = classifier

    return model

model = load_model (arch, hidden_units)

# Recommended to use NLLLoss when using Softmax
criterion = nn.NLLLoss()

# Using Adam optimiser which makes use of momentum to avoid local minima
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# let the model use either GPU or CPU
model.to(device)


# Train the network
start_time = time.time()

epochs = 3
print_every = 10

train_losses = []
validation_losses = []
validation_correct = []
total_train_data = len(dataloaders['train_data'])
total_validation_data = len(dataloaders['valid_data'])

    
# change to cpu/gpu mode
model.to(device)
print(f'Running on: {str(device).upper()}')

'''
**** Train the model ****
'''

# Train the network
start_time = time.time()

epochs = 3
print_every = 10

train_losses = []
validation_losses = []
validation_correct = []
total_train_data = len(dataloaders['train_data'])
total_validation_data = len(dataloaders['valid_data'])
    
# change to cpu/gpu mode
model.to(device)

for i in range(epochs):
    trn_corr = 0
    vld_corr = 0
    
    # Run the training batches
    for b, (inputs, labels) in enumerate(dataloaders['train_data']):
        b+=1
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply the model
        outputs = model(inputs)
        loss = criterion(outputs, labels)
 
        # Tally the number of correct predictions
        predicted = torch.max(outputs.data, 1)[1]
        batch_corr = (predicted == labels).sum()
        trn_corr += batch_corr
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        if b%print_every == 0:
            print(f'epoch: {i:2}  train batch: {b:4} loss: {loss.item():10.8f}')

    train_losses.append(loss)

    # Run the validation batches
    with torch.no_grad():
        for b, (inputs, labels) in enumerate(dataloaders['valid_data']):
                  
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply the model
            y_val = model(inputs)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1] 
            vld_corr += (predicted == labels).sum()
            loss = criterion(y_val, labels)
            
            # Print interim results
            if b%print_every == 0:
                print(f'epoch: {i:2}  validation batch: {b:4} loss: {loss.item():10.8f}  \
accuracy: {vld_corr.item()*100/(10*(b+1)):7.3f}%')
            
            
    validation_losses.append(loss)
    validation_correct.append(vld_corr)

print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed


'''
**** Saving the model ****
'''
# Save the checkpoint 
# feature weights, classifier, class_to_idx mapping
checkpoint = {'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'arch': arch,
              'hidden_units': hidden_units,
              'class_to_idx': image_datasets['train_data'].class_to_idx}

if args.save_dir:
    torch.save(checkpoint, save_dir + '/flower-image-classifier-checkpoint.pth')
else:
    torch.save(checkpoint, 'flower-image-classifier-checkpoint.pth')
    
