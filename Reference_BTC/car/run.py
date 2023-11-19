import cv2
import numpy as np
import yaml
import torch
import time
import json
import os
import math

from lib.utils.utils import (non_max_suppression, get_cfg, letterbox, scale_boxes,
                            detect, pred_road, gstreamer_pipeline)
from lib.utils.plots import box_label, Colors, show_seg_result, show_det_result
from lib.model.trt import trt_model
# from lib.control.UITCar import UITCar


names = ["camtrai", "camphai", "camthang", "trai", "phai", "thang"]

CHECKPOINT = 70
LANEWIGHT = 55            # Độ rộng đường (pixel)
IMAGESHAPE = [160, 80]      # Kích thước ảnh Input model 
def AngCal(image):
    
    h, w = image.shape

    line_row = image[CHECKPOINT, :]
    center = image[h-5, :]
    
    flag = True
    center_min_x = 0
    center_max_x = 0
    
    for x, y in enumerate(center):
        if y == 255 and flag:
            flag = False
            center_min_x = x
        elif y == 255:
            center_max_x = x
            
    center_segment = int((center_max_x+center_min_x)/2)
    x0, y0 = center_segment, h-1
    

    flag = True
    min_x = 0
    max_x = 0
    
    for x, y in enumerate(line_row):
        if y == 255 and flag:
            flag = False
            min_x = x
        elif y == 255:
            max_x = x

    center_row = int((max_x+min_x)/2)
    # gray = cv2.circle(gray, (center_row, CHECKPOINT), 1, 90, 2)
    # cv2.imshow('test', gray)
    
    # x0, y0 = int(w/2), h
    x1, y1 = center_row, CHECKPOINT
    
    image=cv2.line(image,(x1, y1),(x0, y0),(0,0,0),10)

    value = (x1-x0)/(y0-y1)

    angle = math.degrees(math.atan(value))

    # print(steering)
    
    if angle > 60:
        angle = 60
    elif angle < -60:
        angle = -60
    elif angle in range(0,11):
        angle = 0
	# _lineRow = image[CHECKPOINT, :] 
	# count = 0
	# sumCenter = 0
	# centerArg = int(IMAGESHAPE[0]/2)
	# minx=0
	# maxx=0
	# first_flag=True
	# for x, y in enumerate(_lineRow):
	# 	if y == 255 and first_flag:
	# 		first_flag=False
	# 		minx=x
	# 	elif y == 255:
	# 		maxx=x
	 
	# # centerArg = int(sumCenter/count)
	# centerArg=int((minx+maxx)//2)
	# count=maxx-minx

	# # print(minx,maxx,centerArg)
	# # print(centerArg, count)

    # if (count < LANEWIGHT):
    #     if (centerArg < int(IMAGESHAPE[0]/2)):
    #         centerArg -= LANEWIGHT - count
    #     else:
    #         centerArg += LANEWIGHT - count

	# image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

	# _steering = math.degrees(math.atan((centerArg - int(IMAGESHAPE[0]/2))/(IMAGESHAPE[1]-CHECKPOINT)))
	# # print(_steering,"----------",count)
	# image=cv2.line(image,(centerArg,CHECKPOINT),(int(IMAGESHAPE[0]/2),IMAGESHAPE[1]),(255,0,0),1)
    return angle, image


# def control(car, speed, angle):
#     car.setSpeed_cm(speed)
#     car.setAngle(angle)


if __name__ == "__main__":

    # # Motor init
    # Car = UITCar()
    # Car.setMotorMode(0)

    # Setting Camera
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    assert cap.isOpened(), "Camera failed"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    colors = Colors()
    engine, torchscript, imgsize_yolo, imgsize_seg, visualize, save_dir = get_cfg('lib/cfg/cfg.yaml')


    # Loading YOLOv5
    yolo = trt_model(engine)
    yolo = yolo.half()
    yolo.warmup(imgsz=(1, 3, *imgsize_yolo))  # warmup

    
    # Loading Segmentation model
    print(f'Loading {torchscript} for TorchScript inference...')
    extra_files = {'config.txt': ''}  # model metadata
    model_seg = torch.jit.load(torchscript, _extra_files=extra_files, map_location=device)
    model_seg = model_seg.to(device).float()


    while 1:
        ret, frame = cap.read()
        assert ret, "Failed to read camera"

        mask_pred = pred_road(model_seg, frame, imgsize_seg, device)
        if visualize:
            show_seg_result(frame, mask_pred, imgsize_seg, os.path.join(save_dir, "mask.jpg"))

        pred, im = detect(yolo, frame, imgsize_yolo, device)

        for _, det in enumerate(pred):  # per image
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if visualize:
                        show_det_result(frame, xyxy, cls, names, conf, colors, os.path.join(save_dir, "det.jpg"))
            elif visualize:
                cv2.imwrite(os.path.join(save_dir, "det.jpg"), frame)



        


