import sys
sys.path.append('/home/tuan/Desktop/UIT_Car_Racing_2023/sloai_kdong')
from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2
from controller import Controller
import numpy as np
import time
from ultralytics import YOLO
lower_tint = np.array([100, 0, 100])  # Giá trị màu tím thấp
upper_tint = np.array([255, 100, 255])  # Giá trị màu tím cao
check_err = 0
"""OD"""
model_OD = YOLO("best_v8m.pt")
device = 'cuda'
model_OD.to(device)
def time_delay(current_speed, S_OD):
    if (current_speed>52):
        tim_delay = 0.095
    elif (current_speed>51):
        tim_delay = 0.115
    elif (current_speed>50):
        tim_delay = 0.135
    elif (current_speed>49):
        tim_delay = 0.155
    else:
        tim_delay = 0.185
    return tim_delay

i = 0
if __name__ == "__main__":
    try:
        print("Wait......")
        while True:
            state = GetStatus()
            current_speed = state['Speed']
            current_angle = state['Angle']
            raw_image = GetRaw()
            # cv2.imshow("raw", raw_image)
            """ 0: No
                1: turn_right           có
                2: straight             có
                3: no turn left         có
                4: no turn right        có     
                5: no straight
                6: car
                7: unknown
                8: turn left            có
                """
            try:
                results = model_OD.predict(source=raw_image)
                for result in results:
                    result = result.to("cpu")
                    result = result.numpy()
                    xmin, ymin, xmax, ymax = result.boxes.xyxy[0,0], result.boxes.xyxy[0,1], result.boxes.xyxy[0,2], result.boxes.xyxy[0,3]   # box with xyxy format, (N, 4)
                    conf_OD = result.boxes.conf[0]   # confidence score, (N, 1)
                    cls_OD = result.boxes.cls[0]    # cls, (N, 1)
                S_OD = (xmax - xmin)*(ymax - ymin)  # S>1000, nga ba arrmax = 160
            except Exception as er:
                xmax=0
                xmin=0
                ymax=0
                ymin=0
                cls_OD=0
                conf_OD=0
                S_OD=0
                pass
            segment_image = GetSeg()
            if i%10==0:
                cv2.imwrite(f"test/{i}.jpg", segment_image)
            i = i+1
            mask = cv2.inRange(segment_image, lower_tint, upper_tint)
            segment_image[mask > 0] = [255, 255, 255]
            # cv2.imshow('segment_image', segment_image)
            grayscale_image = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)
            # maxspeed = 90, max steering angle = 25
            print("-----------------------------------------------------------------------------------------------------",current_speed, current_angle)
            getControl = Controller(0.12, 0.0000, 0.04, grayscale_image, 124, current_speed, current_angle, check_err, 
                                    cls_OD, conf_OD, xmax, xmin, ymax, ymin, S_OD)
            Control_speed, Control_angle, check_err, right, left, straight, delay = getControl.Control()
            if (right==1 and delay==1):
                print(Control_speed, Control_angle)
                print("--------------------------------------------------==================================--------current_speed", current_speed, current_angle)
                t_delay = time_delay(current_speed, S_OD)
                t_delay2 = 1.33
                if current_angle<-7:
                    angle_straight= 3
                    t_delay = 0
                    t_delay2 + 0.05
                elif current_angle<-1.8:
                    angle_straight= 2.3
                    t_delay = 0
                    t_delay2 + 0.05
                elif current_angle<-1.2:
                    angle_straight= 1.8
                    t_delay = 0
                    t_delay2 + 0.05
                elif current_angle<-0.8:
                    angle_straight= 1.3
                    t_delay = t_delay - 0.03
                    Control_speed=42
                elif current_angle<-0.4:
                    angle_straight= 0.9
                    t_delay = t_delay - 0.025
                    Control_speed=43
                elif current_angle>0.3: 
                    angle_straight= -0.8
                    t_delay = t_delay - 0.02
                    Control_speed=45
                else: angle_straight= 0
                if S_OD>2000 and t_delay>0.04:
                    t_delay = t_delay - 0.04
                    Control_speed=40
                if S_OD>2100:
                    t_delay = 0
                if t_delay==0:
                    Control_speed=40
                t = time.time()
                while (time.time()-t<t_delay): #48-52
                    AVControl(speed=Control_speed, angle=angle_straight)
                    state = GetStatus()
                print("-----------------------------------------------------====================================-----right------current_speed", current_speed, current_angle, S_OD)
                t = time.time()
                while (time.time()-t<t_delay2):
                    AVControl(speed=Control_speed, angle=Control_angle)
                    state = GetStatus()
            elif (left==1 and delay==1):
                print(Control_speed, Control_angle)
                print("-----------------------------------------------======================================----------current_speed", current_speed, current_angle)
                t_delay = time_delay(current_speed, S_OD)
                t_delay2 = 1.33
                if current_angle>7:
                    angle_straight= -3
                    t_delay = 0
                    t_delay2 + 0.05
                elif current_angle>1.8:
                    angle_straight= -2.3
                    t_delay = 0
                    t_delay2 + 0.05
                elif current_angle>1.2:
                    angle_straight= -1.8
                    t_delay = 0
                    t_delay2 + 0.05
                elif current_angle>0.8:
                    angle_straight= -1.3
                    t_delay = t_delay - 0.03
                    Control_speed=42
                elif current_angle>0.4:
                    angle_straight= -0.9
                    t_delay = t_delay - 0.025
                    Control_speed=43
                elif current_angle<-0.3:
                    angle_straight= 0.8
                    t_delay = t_delay - 0.02
                    Control_speed=45
                else: 
                    angle_straight= 0
                if S_OD>2000 and t_delay>0.04:
                    t_delay = t_delay - 0.04
                    Control_speed=40
                if S_OD>2100:
                    t_delay = 0
                if t_delay==0:
                    Control_speed=40
                t = time.time()
                while (time.time()-t<t_delay): #48-52
                    AVControl(speed=Control_speed, angle=angle_straight)
                    state = GetStatus()
                print("-----------------------------------------------===========================================---------left--current_speed", current_speed, current_angle, S_OD)
                t = time.time()
                while (time.time()-t<t_delay2):
                    AVControl(speed=Control_speed, angle=Control_angle)
                    state = GetStatus()
            elif (straight==1 and delay==1):
                print(Control_speed, Control_angle)
                print("-----------------------------------------------======================================----------current_speed", current_speed, current_angle)
                t = time.time()
                while (time.time()-t<0.8): #48-52
                    AVControl(speed=45, angle=Control_angle)
                    state = GetStatus()
            else: 
                AVControl(speed=Control_speed, angle=Control_angle)
            # Lấy kích thước của hình ảnh
            height, width, _ = raw_image.shape

            # Tính kích thước của phần tư cần cắt
            quarter_width = width // 5  # Lấy 1/4 của chiều rộng
            quarter_height = height // 2  # Lấy 1/4 của chiều cao

            # Cắt phần tư ở góc trên phía bên phải
            right_top_quarter = raw_image[:quarter_height, 3 * quarter_width:]
            lower_red = np.array([0, 0, 100])  # Giá trị màu đỏ thấp
            upper_red = np.array([100, 100, 255])  # Giá trị màu đỏ cao
            red_mask = cv2.inRange(right_top_quarter, lower_red, upper_red)

            # Đếm số pixel có màu đỏ
            red_pixel_count = np.count_nonzero(red_mask)
            print("Số pixel có màu đỏ:", red_pixel_count)
            if red_pixel_count>45 and cls_OD!=3 and cls_OD!=4:
                print("stoppppppppppppppppppp")
                AVControl(speed=-100, angle=Control_angle)
            if red_pixel_count>100 and cls_OD!=3 and cls_OD!=4:
                print("stoppppppppppppppppppp now")
                t = time.time()
                while (time.time()-t<2.5): #48-52
                    AVControl(speed=-100, angle=Control_angle)
                    state = GetStatus()
                while True: 
                    AVControl(speed=0, angle=0)
                    state = GetStatus()
            # cv2.imshow('raw_image', right_top_quarter)
            # cv2.waitKey(1)
    finally:
        print('closing socket')
        CloseSocket()
