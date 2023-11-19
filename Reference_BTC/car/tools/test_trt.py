import cv2
import os
import time
import shutil
import numpy as np
import torch
import yaml
import tensorrt as trt
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from collections import OrderedDict, namedtuple
from lib.model.trt import trt_model
from lib.utils.utils import letterbox, non_max_suppression, scale_boxes, detect, gstreamer_pipeline
from lib.utils.plots import box_label, Colors



with open('lib/cfg/cfg.yaml', encoding='ascii', errors='ignore') as f:
    cfg = yaml.safe_load(f)

im_path = cfg['source']
im_size = cfg['imgsize_yolo']
save_dir = cfg['save_dir']
for w in cfg['weights']:
    if w.endswith('.engine'):
        engine = w

names = ["camtrai", "camphai", "camthang", "trai", "phai", "thang"]

def yolo(model, im, im_size, device):
    im = cv2.imread(im)
    imc = np.copy(im)
    st = time.time()
    pred, im = detect(model, im, im_size, device)
    end = time.time()
    print(f"Inference time: {end-st} with {round(1/(end-st))} fps")

    # Visualization
    colors = Colors()
    for _, det in enumerate(pred):  # per image                       
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], imc.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                imc = box_label(imc, xyxy, label, color=colors(c, True))

    return imc



if __name__ == "__main__":

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)
    else:
        os.mkdir(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Loading model
    model = trt_model(engine)
    model.warmup(imgsz=(1, 3, *im_size))  # warmup

    if os.path.isdir(im_path):
        for name in os.listdir(im_path):
            image = os.path.join(im_path, name)
            imc = yolo(model, image, im_size, device)

            # Save the output image
            cv2.imwrite(os.path.join(save_dir, name), imc)
        print(f"Results save to {save_dir}")
        
    elif os.path.isfile(im_path):
        imc = yolo(model, im_path, im_size, device)
        dirname = os.path.dirname(im_path)

        # Save the output image
        print(f"Result save to {dirname}")
        cv2.imwrite(im_path.replace(dirname, save_dir), imc)
    
    else:
       print(f"ERROR: {im_path} doesn't exists")
    


        


