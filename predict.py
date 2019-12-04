"""
@author: Saidul Islam

usage: python predict.py ./flowers/test/100/image_07896.jpg flower-image-classifier-checkpoint.pth --GPU=GPU
"""

from collections import OrderedDict

import numpy as np

from collections import OrderedDict
from math import ceil

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

from PIL import Image

import argparse
import json
import time

# command line arguments for the script
parser = argparse.ArgumentParser ()

parser.add_argument('image_path', help = 'Provide path to image. Mandatory argument', type = str)
parser.add_argument('checkpoint', help = 'Provide path to checkpoint. Mandatory argument', type = str)
parser.add_argument('--arch', help = 'Provide a pretrained network. Currently supported vgg13 and resnet50. Default resnet50.', type = str)
parser.add_argument('--top_k', help = 'Top K most likely classes. Default is 5', type = int)
parser.add_argument('--category_names', help = 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str)
parser.add_argument('--GPU', help = "Option to use GPU. Optional", type = str)


'''
**** handle command line arguments ****
'''
# get command line args for predicting
args = parser.parse_args ()

image_path = args.image_path
checkpoint = args.checkpoint

if args.arch:
    arch = args.arch
else:
    arch = 'resnet50'

#defining number of classes to be predicted. Default = 1
if args.top_k:
    top_k = args.top_k
else:
    top_k = 5
    
#defining device: either cuda or cpu
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'
    
#loading JSON file if provided, else load default file name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    pass

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)


# function to load your checkpoint and rebuild the model
def load_checkpoint(checkpoint, device):
    
    #
    # Checkpoint for when using GPU/CPU
    if device == 'GPU':
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    
    if checkpoint ['arch'] == 'resnet50':
        model = models.resnet50(pretrained = True)
        classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (2048, 1000)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.5)),
                            ('fc2', nn.Linear (1000, checkpoint['hidden_units'])),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.5)),
                            ('fc3', nn.Linear (checkpoint['hidden_units'], 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        model.classifier = classifier
        model.fc = classifier
    else:
        model = models.vgg13(pretrained = True)
        classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, checkpoint['hidden_units'])),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (checkpoint['hidden_units'], 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        model.classifier = classifier
    
    for param in model.parameters():
        param.requires_grad = False
    
    
    model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier'] 
    model.class_to_idx    = checkpoint['class_to_idx']
    
    return model


'''
**** loading the checkpoint ****
'''
# Loading model
model = load_checkpoint(checkpoint, device)



'''
**** image processing ****
'''
def process_image(image):
    
    # Process a PIL image for use in a PyTorch model
    image = F.resize(image, 224)
    
    upper_pixel = (image.height - 224) // 2
    left_pixel = (image.width - 224) // 2
    image = F.crop(image, upper_pixel, left_pixel, 224, 224)
    
    image = F.to_tensor(image)
    image = F.normalize(image, np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
    
    return image


'''
**** predict an image using a trained deep learning model ****
'''
# Implement the code to predict the class from an image file 
def predict(image_path, model, topk, cat_to_name, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image)
    
    with torch.no_grad():
        model.to(device)
        model.eval()
        
        image = image.view(1,3,224,224)
        image = image.to(device)
        
        predictions = model.forward(image)
        
        predictions = torch.exp(predictions)
        top_ps, top_class = predictions.topk(topk, dim=1)
        
        #tensor to np array
        top_ps = np.array(top_ps.cpu())[0]
        top_class = np.array(top_class.cpu())[0]
        
        # Convert to classes
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_class = [idx_to_class[lab] for lab in top_class]
        top_flowers = [cat_to_name[lab] for lab in top_class]
        
    return top_ps, top_class, top_flowers


probs, classes, top_flowers = predict (image_path, model, top_k, cat_to_name, device)


'''
**** print the probabilities ****
'''
for i, j in enumerate(zip(top_flowers, probs)):
    print (f"Rank {i+1}: Flower: {j[0]}, liklihood: {ceil(j[1]*100)}%")
