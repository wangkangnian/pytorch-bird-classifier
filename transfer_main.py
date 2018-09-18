import argparse

from PIL import Image
import torch
import torchvision.transforms as transforms

import sys
sys.path.append('.')
import pretrainedmodels
import pretrainedmodels.utils as utils

model_names = sorted(name for name in pretrainedmodels.__dict__
    if not name.startswith("__")
    and name.islower()
    and callable(pretrainedmodels.__dict__[name]))


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

import dpn

path_img = 'data/hc.jpg'

# Load Model
model = dpn.dpn92()
model.eval()

path_img = path_img
# Load and Transform one input image
load_img = utils.LoadImage()
tf_img = utils.TransformImage(model)

input_data = load_img(path_img) # 3x400x225
input_data = tf_img(input_data)      # 3x299x299
input_data = input_data.unsqueeze(0) # 1x3x299x299
input = torch.autograd.Variable(input_data)

# Load Imagenet Synsets
with open('data/imagenet_synsets.txt', 'r') as f:
    synsets = f.readlines()

# len(synsets)==1001
# sysnets[0] == background
synsets = [x.strip() for x in synsets]
splits = [line.split(' ') for line in synsets]
key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}

with open('data/imagenet_classes.txt', 'r') as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# Make predictions
output = model(input) # size(1, 1000)
max, argmax = output.data.squeeze().max(0)
class_id = argmax
class_key = class_id_to_key[class_id]
classname = key_to_classname[class_key]

print("'{}': '{}' is a '{}'".format('dpn92', path_img, classname))