from pathlib import Path
import torch
import cv2
import numpy as np
import os

from models.yolo import Model
from utils.general import check_requirements, set_logging
from utils.google_utils import attempt_download
from utils.torch_utils import select_device


def custom(model_path='best.pt', autoshape=True):

    model = torch.load(model_path, map_location=torch.device('cpu')) if isinstance(model_path,str) else model_path
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)
    hub_model.load_state_dict(model.float().state_dict())
    hub_model.names = model.names
    if autoshape:
        hub_model = hub_model.autoshape()
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    return hub_model.to(device)


def draw_boxes(model,image, predictions, save_path):
    for i, row in predictions.iterrows():
        xmin, ymin, xmax, ymax, confidence, cls = row[:6]
        label = f'{model.names[int(cls)]} {confidence:.2f}'

        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        c2 = int(xmin) + t_size[0], int(ymin) - t_size[1] - 3
        cv2.rectangle(image, (int(xmin), int(ymin) - t_size[1] - 5), (int(xmin) + t_size[0], int(ymin)), (0, 255, 0),
                      -1, cv2.LINE_AA)

        cv2.putText(image, label, (int(xmin), int(ymin) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, image)


def detect_image(model,image_path, save_dir):
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    model_width, model_height = 640, 480

    img_size = (640, 480)  # modelin giriş boyutu
    image_resized = cv2.resize(image, img_size)

    results = model(image_resized)
    results.print()
    df_prediction = results.pandas().xyxy[0]

    df_prediction['xmin'] = df_prediction['xmin'] * (original_width / model_width)
    df_prediction['xmax'] = df_prediction['xmax'] * (original_width / model_width)
    df_prediction['ymin'] = df_prediction['ymin'] * (original_height / model_height)
    df_prediction['ymax'] = df_prediction['ymax'] * (original_height / model_height)

    save_path = os.path.join(save_dir, os.path.basename(image_path))
    draw_boxes(model,image, df_prediction, save_path)

    return df_prediction


model_path = "./best.pt"
model = custom(model_path=model_path)

images_path = "../yolov7/data/dataset/exp"
save_dir = "../yolov7/data/dataset/exp_detected"

os.makedirs(save_dir, exist_ok=True)

for image_name in os.listdir(images_path):
    image_path = os.path.join(images_path, image_name)
    df_prediction = detect_image(model,image_path, save_dir)
    print(image_name)
