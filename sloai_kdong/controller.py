import numpy as np
import time
import cv2
from collections import Counter
err_arr = np.zeros(10)
pre_t = time.time()
lst_arr = [0,0]
lst_cls = [0,0]
lst_cls2 = [0,0]
class Controller:
    def __init__(self, kp, ki, kd, segment, line, current_speed, current_angle, check_err, cls_OD, conf_OD, xmax, xmin, ymax, ymin, S_OD):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.segment = segment
        self.line = line
        self.current_speed = current_speed
        self.current_angle = current_angle
        self.check_err = check_err
        """OD"""
        self.cls_OD = cls_OD
        self.conf_OD = conf_OD
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin
        self.S_OD = S_OD
        try:
            self.segment = self.remove_small_contours(self.segment)
        except:
            pass
    def Control(self):
        right=0
        delay=0
        left=0
        straight=0
        reset_filter = 0
        straight_detect, straightmin, straightmax = self.Detect_Straight()
        straight_detect2, _, _ = self.Detect_Straight2()
        turnmin, turnmax = self.line_turn()
        line_check_min, line_check_max = self.Line2()
        arr = []
        lineRow = self.segment[self.line,:]
        for x,y in enumerate(lineRow):
            if y==255:
                arr.append(x)
        if not arr:
            if lst_arr[0]>295:
                return 24.9, 20, 1, 0, 0, 0, 0
            if lst_arr[1]<25:
                return -24.9, 20, 1, 0, 0, 0, 0
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
        print(arrmin, arrmax)
        if (straight_detect==1 and straight_detect2==1 and straightmin!=line_check_min and straightmax!=line_check_max and self.cls_OD!=1 and self.cls_OD!=2 and self.cls_OD!=4 and self.cls_OD!=3 and self.cls_OD!=8):
            # and self.cls_OD!=1 and self.cls_OD!=8 and self.cls_OD!=3 and self.cls_OD!=4
            arrmin_lt = self.check(straightmin, 98, line_check_min, 135, self.line)
            arrmax_lt = self.check(straightmax, 98, line_check_max, 135, self.line)
            print(arrmin_lt, arrmax_lt)
            if ((arrmin_lt-arrmin)>22 and abs(arrmax - arrmax_lt)<4 and arrmax<300):
                print(arrmin, arrmin_lt, straightmin, line_check_min)
                print(arrmax, arrmax_lt, straightmax, line_check_max)
                arrmin = int(arrmin_lt)
                print("--------------------------------------min----------------------------nhiễu")
            if ((arrmax - arrmax_lt)>22 and abs(arrmin_lt-arrmin)<4 and arrmin>20):
                print(arrmin, arrmin_lt, straightmin, line_check_min)
                print(arrmax, arrmax_lt, straightmax, line_check_max)
                arrmax = int(arrmax_lt)
                print("---------------------------------------max---------------------------nhiễu")
        center = int((arrmax + arrmin)/2)
        error = int(self.segment.shape[1]/2) - center
        # cv2.circle(self.segment,(arrmin,self.line),5,(0,0,0),3)
        # cv2.circle(self.segment,(arrmax,self.line),5,(0,0,0),3)
        # # cv2.circle(self.segment,(turnmin,123),5,(0,0,0),3)
        # # cv2.circle(self.segment,(turnmax,123),5,(0,0,0),3)
        # cv2.line(self.segment,(center,self.line),(int(self.segment.shape[1]/2),self.segment.shape[0]),(0,0,0),3)
        # cv2.imshow("IMG", self.segment)
        # cv2.waitKey(1)
        angle = -self.PID(error, self.kp, self.ki, self.kd)#0.3
        """OD"""
        if self.filter(reset_filter)==3:
            print("333333333333333333333333333")
            if (self.S_OD>1500):
                right = 1
                self.filter(reset_filter=1)
        if self.filter(reset_filter)==4:
            print("444444444444444444444444444")
            if (self.S_OD>1500):
                left = 1
                self.filter(reset_filter=1)
        if self.filter2(reset_filter)==1:
            print("11111111111111111111111111")
            if (self.S_OD>1500):
                right = 1
                self.filter2(reset_filter=1)
        if self.filter2(reset_filter)==8:
            print("88888888888888888888888888")
            if (self.S_OD>1500):
                left = 1
                self.filter2(reset_filter=1)
        if self.filter2(reset_filter)==2 and straight_detect==1:
            print("22222222222222222222222222")
            if (self.S_OD>2000):
                straight = 1
                self.filter2(reset_filter=1)
        if (self.conf_OD>0.8 and self.xmax<315 and self.ymin > 5 and self.cls_OD==1 and self.S_OD>1500):
            right = 1
            self.filter2(reset_filter=1)
        if (self.conf_OD>0.8 and self.xmax<315 and self.ymin > 5 and self.cls_OD==3 and self.S_OD>1500):
            right = 1
            self.filter(reset_filter=1)
        if (self.conf_OD>0.8 and self.xmax<315 and self.ymin > 5 and self.cls_OD==4 and self.S_OD>1500):
            left = 1
            self.filter(reset_filter=1)
        if (self.conf_OD>0.8 and self.xmax<315 and self.ymin > 5 and self.cls_OD==8 and self.S_OD>1500):
            left = 1
            self.filter2(reset_filter=1)
        """speed"""
        if (right==1):
            angle, speed, delay = self.Turn_Right(turnmin, turnmax)
            if angle==0:
                speed = self.Speed_control(angle, error, 1)
        elif (left==1):
            angle, speed, delay = self.Turn_Left(turnmin, turnmax)
            if angle==0:
                speed = self.Speed_control(angle, error, 1)
        elif (straight==1):
            angle, speed, delay = self.Go_straight(turnmin, turnmax)
            if angle==50:
                speed = self.Speed_control(angle, error, 1)
        else:
            if (self.conf_OD>0.6 and self.S_OD>400 and self.cls_OD!=0 and self.cls_OD!=6 and self.cls_OD!=7):
                if (self.cls_OD==1 or self.cls_OD==3 or self.cls_OD==4 or self.cls_OD==8):
                    speed = self.Speed_control(angle, error, 1)
                else:
                   speed = self.Speed_control(angle, error, 0) 
            else: 
                speed = self.Speed_control(angle, error, 0)
            if self.current_speed > 44:
                if angle>21:
                    angle = 24.9
                if angle<-21:
                    angle =-24.9
            if abs(self.current_angle-angle)>8.5 and self.current_speed>44:
                return 0, angle, 0, 0, 0, 0, 0
        return speed, angle, 0, right, left, straight, delay
    """OD"""
    def filter(self, reset_filter):
        if (self.S_OD>600 and self.conf_OD>0.75):
            if (self.cls_OD==3 or self.cls_OD==4):
                lst_cls.append(self.cls_OD)
        if reset_filter==1:
            for i in range(len(lst_cls)):
                lst_cls[i] = 0
        if len(lst_cls) > 3:
            lst_cls.pop(0)
        try:
            count = Counter(lst_cls)
            most_common_element = count.most_common(1)[0]
            return most_common_element[0]
        except:
            return 0
    def filter2(self, reset_filter): 
        if (self.S_OD>600 and self.conf_OD>0.8):
            if (self.cls_OD==1 or self.cls_OD==8 or self.cls_OD==2):
                lst_cls2.append(self.cls_OD)
        if reset_filter==1:
            for i in range(len(lst_cls2)):
                lst_cls2[i] = 0
        if len(lst_cls2) > 3:
            lst_cls2.pop(0)
        try:
            count = Counter(lst_cls2)
            most_common_element = count.most_common(1)[0]
            return most_common_element[0]
        except:
            return 0
    def line_turn(self):
        turnarr = []
        line_turn = self.segment[123,:]
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
    def Turn_Right(self, turnmin, turnmax):
        if turnmax>290:
            angle=24.9
            speed = 43
            delay = 1
        else:
            angle = 0
            speed = 48
            delay = 0
        return angle, speed, delay
    def Turn_Left(self, turnmin, turnmax):
        if turnmin<30:
            angle=-24.9
            speed = 43
            delay = 1
        else:
            angle = 0
            speed = 48
            delay = 0
        return angle, speed, delay
    def Go_straight(self, turnmin, turnmax):
        if turnmin<30 or turnmax>290:
            if abs(self.current_angle)>0.5:
                angle = -self.current_angle/3
                speed = 40
                delay = 1
            else:
                angle = 0
                speed = 40
                delay = 1
        else:
            angle = 50
            speed = 40
            delay = 0
        return angle, speed, delay
    """Segment"""
    def Speed_control(self, angle, error, set_speed_od):
        # if abs(angle)<3:
        #     set_speed = 56 - abs(error)/5
        # elif abs(angle)<6:
        #     set_speed = 55 - abs(error)/3.5
        # elif abs(angle)<9:
        #     set_speed = 54 - abs(error)/3.5
        # else:
        #     set_speed = 53 - abs(error)/3.5
        if abs(angle)<3:
            set_speed = 55- abs(error)/5
        elif abs(angle)<6:
            set_speed = 55 - abs(error)/4
        elif abs(angle)<9:
            set_speed = 55 - abs(error)/3.8
        else:
            set_speed = 54 - abs(error)/4
        if self.cls_OD==6 and self.conf_OD>0.65:
            set_speed = 52
        if (set_speed_od==1):
            if (self.cls_OD==1 or self.cls_OD==8):
                set_speed = 49.5
            else:
                set_speed = 49.5
        if (float(self.current_speed)<set_speed):
            if abs(float(self.current_speed)-set_speed)<2:
                if (set_speed_od==1):
                    speed = -1.5*(3-abs(float(self.current_speed)-set_speed)) + 90
                else:
                    speed = -2.2*(3-abs(float(self.current_speed)-set_speed)) + 90
            elif abs(float(self.current_speed)-set_speed)<4:
                speed = 80
            else:
                speed = 90
        else: 
            if (abs(angle)>3.5):
                speed = 0
            else:
                if (set_speed_od==1):
                    speed = 0
                else:
                    speed = 35
        if float(self.current_speed)<35 and set_speed_od==0:
            speed = 90
        if float(self.current_speed)>45 and abs(angle)>20:
            speed = -12
            print("============================================================brake")
        return speed
    def check(self,a,b,c,d,y):
        phi1 = (b-d)/(a-c)
        phi2 = b - phi1*a
        x = (y-phi2)/phi1
        return x
    def Detect_Straight(self):
        straightarr = []
        lineStraight = self.segment[98,:]
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
    def Detect_Straight2(self):
        straightarr = []
        lineStraight = self.segment[96,:]
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
    def Line2(self):
        linearrr = []
        Linearr = self.segment[135,:]
        for x,y in enumerate(Linearr):
            if y==255:
                linearrr.append(x)
        try: 
            linemax=max(linearrr)
            linemin=min(linearrr)
        except Exception as er:
            linemin=0
            linemax=0
            pass
        return linemin, linemax
    def PID(self, err, Kp, Ki, Kd):
        global pre_t
        err_arr[1:] = err_arr[0:-1]
        err_arr[0] = err
        delta_t = time.time() - pre_t
        pre_t = time.time()
        P = Kp*err
        D = Kd*(err - err_arr[1])/delta_t
        I = Ki*np.sum(err_arr)*delta_t
        angle = P + I + D
        return angle
    def remove_small_contours(self, image):
        image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
        image_remove = cv2.bitwise_and(image, image, mask=mask)
        return image_remove