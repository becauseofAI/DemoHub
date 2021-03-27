# -*- coding: utf-8 -*-
import cv2
import torch
from lfd.execution.utils import load_checkpoint
from lfd.data_pipeline.augmentation import *

# set the target model script
from deploy.WIDERFACE_LFD_L import config_dict, prepare_model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

prepare_model()

# set the model weight file
param_file_path = './deploy/lfd_l_face_20210210_10423_epoch_1000.pth'

# load the model weight file
load_checkpoint(config_dict['model'], load_path=param_file_path, strict=True)

# set the image path to be tested
image_path = './images/image1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# model predict
results = config_dict['model'].predict_for_single_image(image, aug_pipeline=simple_widerface_val_pipeline, classification_threshold=0.5, nms_threshold=0.3)
# results: [[class_id, confidence, x_left, y_top, width, height], [...]]
print(results)

# plot predicted bounding box
for bbox in results:
    # print(bbox)
    cv2.rectangle(image, (int(bbox[2]), int(bbox[3])), (int(bbox[2] + bbox[4]), int(bbox[3] + bbox[5])), (0, 255, 0), 1)
print('predect device: %s' % device.type)
print('predect faces num: %d' % len(results))

# save and show the result images
cv2.imwrite('results/' + image_path.split('/')[-1], image)
cv2.imshow('result', image)
print('Press ESC to exit the program normally.')
cv2.waitKey()
cv2.destroyWindow('result')
