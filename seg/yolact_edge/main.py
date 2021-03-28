import torch
import logging
from data import set_cfg
from models.yolact import Yolact
from deploy.predict import parse_args, predict
from utils.functions import SavePath
from utils.logging_helper import setup_logger


args = parse_args()

cuda = False
if torch.cuda.is_available():
    cuda = True

weight_path = 'weights/yolact_edge_mobilenetv2_54_800000.pth'
image_path = 'images/demo.jpg'
result_path = 'results/demo.png'

model_path = SavePath.from_str(weight_path)
# TODO: Bad practice? Probably want to do a name lookup instead.
config = model_path.model_name + '_config'
print('Config not specified. Parsed %s from the file name.\n' % config)
set_cfg(config)

setup_logger(logging_level=logging.INFO)
logger = logging.getLogger("yolact.eval")

logger.info('Loading model...')
model = Yolact(training=False)
model.load_weights(weight_path, args=args)
model.eval()
logger.info('Model loaded.')

if cuda:
    model = model.cuda()
    logger.info('Predicting with gpu.')
else:
    logger.info('Predicting with cpu.')

predict(image_path, result_path, model, score_threshold=0.3, top_k=100, cuda=cuda, logger=logger)