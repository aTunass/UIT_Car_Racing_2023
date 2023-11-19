import cv2
import os
import shutil
import numpy as np
import torch
import yaml
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.utils.utils import pred_road, 
from lib.utils.plots import show_seg_result


with open('lib/cfg/cfg.yaml', encoding='ascii', errors='ignore') as f:
    cfg = yaml.safe_load(f)

im_path = cfg['source']
im_size = cfg['imgsize_seg']
save_dir = cfg['save_dir']
for w in cfg['weights']:
    if w.endswith('.torchscript'):
        torchscript = w


def detect(model, im, device):
    im = cv2.imread(im)

    # Preprocessing
    st = time.time()
    mask_pred = pred_road(model, im, im_size, device)
    
    end = time.time()

    print(f"Inference time: {end-st} with {round(1/(end-st))}")

    return mask_pred, im



if __name__ == "__main__":

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)
    else:
        os.mkdir(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Loading model
    print(f'Loading {torchscript} for TorchScript inference...')
    extra_files = {'config.txt': ''}  # model metadata
    model = torch.jit.load(torchscript, _extra_files=extra_files, map_location=device)
    model = model.to(device).float()

    # Warmup
    for _ in range(3):
        image = torch.randn((1, 3, *im_size)).to(device).float()
        out = model(image)

    if os.path.isdir(im_path):
        for name in os.listdir(im_path):
            image = os.path.join(im_path, name)
            mask_pred, im = detect(model, image, device)

            # Save the output image
            show_seg_result(im, mask_pred, im_size, os.path.join(save_dir, name))
        print(f"Results save to {save_dir}")

    elif os.path.isfile(im_path):
        mask_pred, im = detect(model, im_path, device)
        dirname = os.path.dirname(im_path)

        # Save the output image
        show_seg_result(im, mask_pred, im_size, im_path.replace(dirname, save_dir))
        print(f"Result save to {dirname}")
        cv2.imwrite(im_path.replace(dirname, save_dir), im)

    else:
       print(f"ERROR: {im_path} doesn't exists")
    


        


