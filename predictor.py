from io import BytesIO
import torch
from torch import argmax, load
from torch import device as DEVICE
from torch.cuda import is_available
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from PIL import Image
from torchvision import transforms, models
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models import resnet50
from flask import Flask, jsonify, request

app = Flask(__name__)
LABELS = ['None', 'Meningioma', 'Glioma', 'Pitutary']

device = "cuda" if is_available() else "cpu"

resnet_model = resnet50(pretrained=True)

for param in resnet_model.parameters():
    param.requires_grad = True

n_inputs = resnet_model.fc.in_features
resnet_model.fc = Sequential(Linear(n_inputs, 2048),
                            SELU(),
                            Dropout(p=0.4),
                            Linear(2048, 2048),
                            SELU(),
                            Dropout(p=0.4),
                            Linear(2048, 4),
                            LogSigmoid())

for name, child in resnet_model.named_children():
    for name2, params in child.named_parameters():
        params.requires_grad = True

resnet_model.to(device)
resnet_model.load_state_dict(load('./model/bt_resnet50_model.pt', map_location=DEVICE(device)))
resnet_model.eval()

import numpy as np
from keras.utils import image_utils as image
from keras.models import load_model
saved_model = load_model("./model/model.h5")

transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
LABELS = ['None', 'Meningioma', 'Glioma', 'Pitutary']
def check(input_img):
    print(" your image is : " + input_img)
    img = image.load_img("images/" + input_img, target_size=(256, 256))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    output = saved_model.predict(img)

    print(output)
    if output[0][0] == 1:
        del(img)
        return 'Healthy'
    else:
        with torch.no_grad():
            img = Image.open("images/" + input_img)
            img = transform(img).to(device)
            img = img[None, ...]
            y_hat = resnet_model.forward(img.to(device))
            predicted = torch.argmax(y_hat.data, dim=1)
            del(img)
            return (LABELS[predicted.data])
