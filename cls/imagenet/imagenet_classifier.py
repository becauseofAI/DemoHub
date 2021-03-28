import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()


def transform_image(image_path):
    transforms_compose = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                                 [0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])
    image = Image.open(image_path)
    return transforms_compose(image).unsqueeze(0)


def get_prediction(image_path):
    tensor = transform_image(image_path=image_path)
    outputs = model.forward(tensor)
    results = torch.softmax(outputs, dim=1).detach().numpy()[0]
    pred_idx = np.argmax(results)
    pred_prob = np.max(results)
    class_id, class_name = imagenet_class_index[str(pred_idx)]
    return {'class_id': class_id, 'class_name': class_name, 'class_prob': pred_prob}
