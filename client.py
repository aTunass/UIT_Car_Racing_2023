from lib.control.UITCar import UITCar

import pytimedinput
import Jetson.GPIO as GPIO
import math


import os
import time
import argparse
import numpy as np
import onnxruntime
import sys
import cv2
import time
from controller import Controller, road_lines, remove_small_contours, find_line
import tensorrt as trt
from lib.model.trt import trt_model
import torch
from lib.utils.utils import letterbox, non_max_suppression, scale_boxes, detect
def gstreamer_pipeline(
    capture_width=640,
    capture_height=320,
    display_width=640,
    display_height=320,
    framerate=20,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
if __name__ == "__main__":
    Car = UITCar()
    Car.Motor_ClearErr()
    Car.setAngle(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    assert cap.isOpened(), "Camera failed"
    model_OD = trt_model('weights/best_new.engine')
    model_OD.warmup(imgsz=(1, 3,*(640,640)))
    # ["no_straight", "turn_right", "turn_left", "straight", "no_turn_left", "no_turn_right", "stop", "traffic_lights"]
    session_lane = onnxruntime.InferenceSession('weights/model-080.onnx', None, providers=['CPUExecutionProvider'])
    input_name_lane = session_lane.get_inputs()[0].name
    i = 0
    j = 0
    left = 0
    right = 0
    straight = 0
    lefturn = 0
    rightturn = 0
    straighturn = 0
    stop = 0
    stop_now = 0
    index = 2
    traffic_signal = 0
    pre_index = 2
    while 1:
        start = time.time()
        ret, frame = cap.read()
        frame_OD = np.copy(frame)
        OD_check=0
        pred = 0
        """segment"""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = frame_gray[100:, :]
        img = cv2.resize(img, (160,40))
        current_angle = Car.getAngle()
        img = np.expand_dims(img, 2)
        DAsegmentation = road_lines(img, session=session_lane, inputname=input_name_lane) 
        try:
            DAsegmentation = remove_small_contours(DAsegmentation)
        except:
            DAsegmentation = DAsegmentation
        """OD"""
        if (i%4==0):
            pred, im = detect(model_OD, frame_OD, (640,640), device)
            OD_check = 1
            i=0
            print('----------------------------------------pred ', pred)
        getControl = Controller(1, 0.04, DAsegmentation, 20, OD_check, current_angle, pred, left, right, straight, frame, lefturn, straighturn, rightturn, stop, stop_now, index)
        Control_speed, Control_angle, left, right, straight, lefturn, straighturn, rightturn, stop, stop_now, index, start= getControl.Control()
        if (index>pre_index):
            traffic_signal = traffic_signal + 1
        pre_index = index
        print("i-----------------------------------------------------------------------------------------------------------------index", index)
        if rightturn==1:
            right = 0
        if index==2:
            if lefturn==1:
                print("-------------------------------------------------------------------------------------turn left", Control_angle, Control_speed)
                print(left)
                if traffic_signal==0:
                    Car.setAngle(13)
                    Car.setSpeed_rad(23)#20
                    t = time.time()
                    while(time.time()-t<2.25):
                        ret_c, frame_c = cap.read()
                    lefturn = 0
                    left = 0
                else:
                    traffic_signal = 0
                    Car.setAngle(15)
                    Car.setSpeed_rad(23)#20
                    t = time.time()
                    while(time.time()-t<2.25):
                        ret_c, frame_c = cap.read()
                    lefturn = 0
                    left = 0
            elif straighturn==1:
                print("-------------------------------------------------------------------------------------straight", Control_angle, Control_speed)
                t = time.time()
                Car.setAngle(0)
                Car.setSpeed_rad(25)
                straighturn = 0
                straight = 0
            elif stop_now==1:
                Car.setAngle(0)
                Car.setSpeed_rad(-6)
                t = time.time()
                while(time.time()-t<1.5):
                    ret_c, frame_c = cap.read()
                Car.setAngle(0)
                Car.setSpeed_rad(0)
                while(time.time()-t<0.5):
                    ret_c, frame_c = cap.read()
                Car.setAngle(0)
                Car.setSpeed_rad(-1)
                while(time.time()-t<0.5):
                    ret_c, frame_c = cap.read()
                print("stopnow")
                stop_now = 0
                stop = 0
                while(1):
                    Car.setSpeed_rad(-1)
                    # Car.setSpeed_rad(0)
                    ret_c, frame_c = cap.read()
            else:
                Car.setAngle(Control_angle)
                Car.setSpeed_rad(Control_speed)
        else:
            Car.setAngle(0)
            Car.setSpeed_rad(-5)
            t =time.time()
            while(time.time()-t<1.2):
                ret_c, frame_c = cap.read()
            Car.setAngle(0)
            Car.setSpeed_rad(0)
        i = i+1
        end = time.time()
        print("fps", 1/(end-start))