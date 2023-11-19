import time
import numpy as np
import cv2
from lib.control.UITCar import UITCar
pre_t = time.time()
err_arr = np.zeros(7)
lst_arr = [0,0]
# signal = [2, 1, 5, 0, 2, 4, 3, 6]
signal = [2]
def remove_small_contours(image):
    image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
    image_remove = cv2.bitwise_and(image, image, mask=mask)
    return image_remove
def find_line(image):
    image = image[200:]
    image = cv2.resize(image, (160, 80))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 155], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    result = cv2.bitwise_and(image, image, mask=mask_white)
    _, binary_result = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)
    return binary_result   
def road_lines(image, session, inputname):
	# Crop ảnh lại, lấy phần ảnh có làn đườngs
	small_img = image/255
	small_img = np.array(small_img, dtype=np.float32)
	small_img = small_img[None, :, :, :]
	prediction = session.run(None, {inputname: small_img})
	prediction = np.squeeze(prediction)
	prediction = np.where(prediction < 0.5, 0, 255)
	# prediction = prediction.reshape(small_img.shape[0], small_img.shape[1])
	prediction = prediction.astype(np.uint8)
	return prediction
def line_ver(img, l_ver):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arr_ver = []
    line_ver = img[:,l_ver]
    for x,y in enumerate(line_ver):
        if y==255:
            arr_ver.append(x)
    try:
        max_ver=max(arr_ver)
        if (70>max_ver and max_ver>50):
            ver_detect = 1
        else: ver_detect = 0
        # print('left', left_detect)
    except Exception as er:
        ver_detect = 0
        pass
    return ver_detect
def detect_hor(img):
    return line_ver(img, 40) + line_ver(img, 60) + line_ver(img, 80) + line_ver(img, 100) + line_ver(img, 120)
class Controller:
    def __init__(self, kp, kd, segment, line, OD_check, current_angle, pred, left, right, straight, frame, leftturn, straightturn, rightturn, stop, stop_now, index):
        self.kp = kp
        self.kd = kd
        self.segment = segment
        self.line = line
        self.OD_check = OD_check
        self.frame = frame
        self.leftturn = leftturn
        self.straightturn = straightturn
        self.rightturn = rightturn
        self.stop = stop
        self.stop_now = stop_now
        self.index = index
        try:
            self.cls, self.conf, self.box, self.traffic_cls, self.traffic_conf, self.traffic_box = self.box_conf(pred)
            self.traffic_S = (self.traffic_box[2]-self.traffic_box[0])*(self.traffic_box[3]-self.traffic_box[1])
            self.S = (self.box[2]-self.box[0])*(self.box[3]-self.box[1])
        except Exception as erorrrr:
            print(erorrrr)
            self.cls, self.conf, self.box, self.traffic_cls, self.traffic_conf, self.traffic_box = 0, 0, [0,0,0,0], 0, 0, [0,0,0,0]
            self.S = (self.box[2]-self.box[0])*(self.box[3]-self.box[1])
            self.traffic_S = (self.traffic_box[2]-self.traffic_box[0])*(self.traffic_box[3]-self.traffic_box[1])
        print(self.cls, self.conf, self.box, self.traffic_cls, self.traffic_conf, self.traffic_box)
        self.current_angle = current_angle
        self.LANEWIGHT=77
        self.width=0
        self.left=left
        self.right=right
        self.straight=straight
    def Control(self):
        start = 2
        #print("-----------------------------------------OD ", self.cls, self.conf, self.S)
        turnmin, turnmax = self.line_turn()
        arr2min, arr2max = self.line_2()
        straight_detect, straightmin, straightmax = self.Detect_Straight()
        #print("line turn", turnmin, turnmax)
        intersection = 0
        arr = []
        lineRow = self.segment[self.line,:]
        for x,y in enumerate(lineRow):
            if y==255:
                arr.append(x)
        if not arr:
            if lst_arr[0]>155:
                print("error_right", self.current_angle,)
                return 10,self.current_angle, 0, 0, 0, 0, 0, 0, 0, 0, self.index, 2
            if lst_arr[1]<20:
                print("error_left", self.current_angle,)
                return 10,self.current_angle, 0, 0, 0, 0, 0, 0, 0, 0, self.index, 2
        try: 
            arrmax=max(arr)
            lst_arr.append(arrmax)
            if len(lst_arr) > 2:
                lst_arr.pop(0)
            arrmin=min(arr)
            lst_arr.append(arrmin)
            if len(lst_arr) > 2:
                lst_arr.pop(0)
        except:
            arrmax = lst_arr[0]
            arrmin = lst_arr[1]
        if self.rightturn==1:
            arrmax = arr2max
        if arrmin<(arrmax-64):
            arrmin = arrmax - 64
        else:
            arrmin = arrmin
        if arrmax>145:
            if arrmin<(arrmax-65):
                arrmin = arrmax - 65
            else:
                arrmin = arrmin
        center = int((arrmax + arrmin)/2)
        error = int(self.segment.shape[1]/2) - center
        angle = self.PID(error, self.kp, self.kd)#0.3
        """OD"""
        if self.traffic_cls==7 and self.traffic_conf>0.6 and self.traffic_S>1500:
            print("traffic", self.traffic_S)
            self.index = self.traffic_light_det(self.frame, self.traffic_box)
            print(self.index)
        elif self.cls==7 and self.conf>0.8 and self.S>1500:
            print("traffic 2", self.S)
            start= self.traffic_light_det(self.frame, self.box)
        elif self.cls==6 and self.conf>0.7 and self.S>500:
            self.stop=1
        elif self.conf > 0.8 and self.S>1500:
            if self.cls==2:
                self.left=1
            elif self.cls==1 and self.left==0 and self.straight==0:
                self.right=1
            elif self.cls==4 and self.left==0 and self.straight==0:
                self.right=1
            elif self.cls==0:
                self.left=1
            elif self.cls==3 and self.right==0 and self.left==0:
                self.straight=1
        #print("signal", self.left)
        if self.left==1:
            self.leftturn = self.turn_left(turnmin)
        elif self.right==1:
            self.rigth=0
            self.rightturn = self.turn_right()
        elif self.straight==1:
            self.go_straight(turnmin, turnmax)
        elif self.stop==1:
            self.stop_now = self.stopppppp(turnmin)
        speed = self.speed_control(error)
        if angle>18:
            angle = angle+1
        if angle<-18:
            angle = angle - 1
        return speed, angle, self.left, self.right, self.straight, self.leftturn, self.straightturn, self.rightturn, self.stop, self.stop_now, self.index, start
    def traffic_light_det(self, img, box): # boxes: x,y,w,h
        img = cv2.resize(img, (640,640))
        '''
        input: img, box
        output: 0: red, 1: yellow, 2: green
        '''
        # TODO: Check boxes
        GHSVLOW = np.array([45, 100, 100])      #green
        GHSVHIGH = np.array([90, 255, 255])  
        YHSVLOW = np.array([20, 100, 100])      #yellow
        YHSVHIGH = np.array([40, 255, 255])     
        RHSVLOW = np.array([160,100,100])       #red 
        RHSVHIGH = np.array([180,255,255])      
        RHSVLOW_1 = np.array([0,70,50])         #red2
        RHSVHIGH_1 = np.array([10,255,255])
        # TODO: Check boxes
        x1, y1 = box[:2]
        x2, y2 = box[2:]
        img_crop = img[int(y1):int(y2), int(x1):int(x2)]
        img_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        maskg = cv2.inRange(img_hsv, GHSVLOW, GHSVHIGH)
        masky = cv2.inRange(img_hsv, YHSVLOW, YHSVHIGH)
        maskr_1 = cv2.inRange(img_hsv, RHSVLOW, RHSVHIGH)
        maskr_2 = cv2.inRange(img_hsv, RHSVLOW_1, RHSVHIGH_1)
        maskr = maskr_1 | maskr_2

        area = [self.check(mask) for mask in [maskr, masky, maskg]]
        index = area.index(max(area))
        return index
    def check(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return 0
        max_contour = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(max_contour)
        return area
    def box_conf(self, pred):
        for _, det in enumerate(pred):  # per image                
            if len(det)>0 and len(det)<2:
                print("1")
                return float(det[0][5]), float(det[0][4]), list(det[0][:4].tolist()), 0, 0, [0,0,0,0]
            elif len(det)>1:
                your_list = det.tolist()
                has_value_7 = any(7.0 in sublist for sublist in your_list)
                if has_value_7:
                    rows_with_7 = [sublist for sublist in your_list if 7.0 in sublist]
                    rows_without_7 = [sublist for sublist in your_list if 7.0 not in sublist]
                    print("Các dòng có giá trị 7.0:", rows_with_7)
                    return rows_without_7[0][5], rows_without_7[0][4], rows_without_7[0][:4], rows_with_7[0][5], rows_with_7[0][4], rows_with_7[0][:4]
                else:
                    return float(det[0][5]), float(det[0][4]), list(det[0][:4].tolist()), 0, 0, [0,0,0,0]
            else:
                return 0, 0, [0,0,0,0], 0, 0, [0,0,0,0]
        return 0, 0, [0,0,0,0], 0, 0, [0,0,0,0]
    def Detect_Straight(self):
        straightarr = []
        lineStraight = self.segment[25,:]
        for x,y in enumerate(lineStraight):
            if y==255:
                straightarr.append(x)
        try: 
            straightmax=max(straightarr)
            straightmin=min(straightarr)
            straight_detect = 1
        except Exception as er:
            straight_detect = 0
            straightmin = 0
            straightmax = 0
            pass
        return straight_detect, straightmin, straightmax
    def stopppppp(self, turnmin):
        # binary = find_line(self.frame)
        if self.S>3000:
            return 1
        else:
            return 0
    def turn_left(self, turnmin):
        binary = find_line(self.frame)
        if detect_hor(binary)>3:
            return 1
        else:
            return 0
    def turn_right(self):
        binary = find_line(self.frame)
        if detect_hor(binary)>3:
            return 1
        else:
            return 0
        # return 1
    def go_straight(self, turnmin, turnmax):
        binary = find_line(self.frame)
        if detect_hor(binary)>3:
            if turnmin<20 and turnmax<150:
                self.straightturn=0
            else:
                self.straightturn=1
        else:
            self.straightturn=0
        return 0
    def speed_control(self, error):
        # print("err", error)
        if self.OD_check==1:
            return int(-0.18*abs(error) + 18)
        elif self.rightturn==1:
            return int(-0.18*abs(error) + 20)
        else:
            return int(-0.19*abs(error) + 25)
    def line_turn(self):
        turnarr = []
        line_turn = self.segment[19,:]
        for x,y in enumerate(line_turn):
            if y==255:
                turnarr.append(x)
        try: 
            turnmax=max(turnarr)
            turnmin=min(turnarr)
        except Exception as er:
            turnmin = 0
            turnmax = 0
        return turnmin, turnmax
    def line_2(self):
        arr2 = []
        arr2_turn = self.segment[18,:]
        for x,y in enumerate(arr2_turn):
            if y==255:
                arr2.append(x)
        try: 
            arr2max=max(arr2)
            arr2min=min(arr2)
        except Exception as er:
            arr2min = 0
            arr2max = 0
        return arr2min, arr2max
    def PID(self, err, Kp, Kd):
        global pre_t
        err_arr[1:] = err_arr[0:-1]
        err_arr[0] = err
        delta_t = time.time() - pre_t
        pre_t = time.time()
        P = Kp*err
        D = Kd*(err - err_arr[1])/delta_t
        angle = P + D
        if angle >= 32:
            angle = 32
        elif angle <= -32:
            angle = -32
        return int(angle)