import cv2
import json
import numpy as np
from deploy.predict import predict
from models.experimental import attempt_load
from utils.torch_utils import select_device


cin = open('./cfg/config.json', 'r', encoding='utf8')
cfg = json.load(cin)
# select device
device = select_device(cfg['device'])
# load model
model = attempt_load(cfg['weights'], map_location=device)

image_path = 'images/dog.jpg'

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_arr = np.array(img)

# model predict
results = predict(cfg, model, img_arr)
print('json results:\n', results)