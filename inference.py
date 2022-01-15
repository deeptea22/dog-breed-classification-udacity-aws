import os
import io
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


CLASSES_NUMBER = 133

'''
This file includes the required methods (model_fn to load the model and input_fn to transform the input into something which can be understood by the model) for the model to be deployed.
'''
def model_fn(model_dir):
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, CLASSES_NUMBER))

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model


def input_fn(request_body, content_type):
    image = Image.open(io.BytesIO(request_body))

    transformation = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor()])

    return transformation(image).unsqueeze(0)