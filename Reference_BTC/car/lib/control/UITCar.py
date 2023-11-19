#-*- coding: utf-8 -*-
import string
import numpy as np
from adafruit_servokit import ServoKit
from threading import Thread
import threading
import time

import serial
import socket

from pathlib import Path
from PIL import Image

import luma.oled.device as OledDevice
from luma.core.render import canvas
import Jetson.GPIO as GPIO


#global variable
Car_MaxAngle = 90
Car_MaxSpeed_rad = 60
Car_MaxSpeed_cm = 72
ServoChannel = 0

_but1_pin = "UART2_RTS"  #17
_but2_pin = "SPI2_SCK"   #27
_but3_pin = "LCD_TE"     #22

_ServoOEPin = "DAP4_SCLK"
class UITCar:
    #private variable:
    __kit = ServoKit(channels=16)   #servo
    __serial_port = serial.Serial() #motor
    __device = OledDevice.sh1106()  #oled
    __BTN = [_but1_pin,_but2_pin,_but3_pin]
    __OLED_text = ['', '', '', '','']
    __OLED_line = [3, 4, 5, 6, 7]
    __OLED_left = [1, 1, 1, 1, 1]
    __is_Print  = [-1, -1, -1, -1, -1]
    __vel   =   0
    __dong  =   0
    __err   = ""
    __MotorMode = 0
    
    def __init__(self) :
        self.__Button_Init()
        #self.__Oled_Init()
        self.__Servo_Init()
        self.__Motor_Init()

    def __del__(self) :
        print("Del object")
        self.setAngle(0)
        self.setSpeed_rad(0)
        self.__ThreadIsKill = True
        
    def __Oled_Init(self, contrast = 255):
        """- Chức năng: Khởi tạo cho màn hình OLED \n
        - Biến: \n
            + Contrast: Độ sáng cho màn hình từ 0->255\n
        - Trả về: None
        """
        self.__device.contrast(contrast)
        self.__OLED_Logo()
        print("OLED INIT")

    def __Servo_Init(self):
        """- Chức năng: Khởi tạo cho Servo\n
        - Biến: None\n
        - Trả về: None
        """
        GPIO.setup(_ServoOEPin, GPIO.OUT)
        GPIO.output(_ServoOEPin, GPIO.LOW) 
        self.__kit.servo[ServoChannel].set_pulse_width_range(min_pulse=600, max_pulse=2400)
        kitt = self.__kit
        self.setAngle(0)
        return kitt
    
    def __Button_Init(self):
        """- Chức năng: Thực hiện khởi tạo 3 nút nhấn 
            -Biến:None
            -Trả về:None
        """
        GPIO.setup(self.__BTN, GPIO.IN)
        print("Done Button setup")
        
        GPIO.add_event_detect(self.__BTN[0], GPIO.FALLING, self._Button1_Pressed)
        GPIO.add_event_detect(self.__BTN[1], GPIO.FALLING, self._Button2_Pressed)
        GPIO.add_event_detect(self.__BTN[2], GPIO.FALLING, self._Button3_Pressed)

    
    def __Motor_Init(self):
        self.__serial_port.port="/dev/ttyTHS1"
        self.__serial_port.baudrate=115200
        self.__serial_port.bytesize=serial.EIGHTBITS
        self.__serial_port.parity=serial.PARITY_NONE
        self.__serial_port.stopbits=serial.STOPBITS_ONE
        self.__serial_port.timeout=1
        self.__serial_port.open()

        self.__motorChangeControl()
        self.__motorDisableProtect()
        self.__motorUnlock()
        self.setMotorMode(0)
        self.__motorSaveRestart()
        time.sleep(5)

        self.__motorNoACK()
        self.__motorUnlock()
        self.setSpeed_rad(0)        
        print("Motor Setup Done")
        thr = threading.Thread(target=self.thr_motor_recv)
        thr.start()

    def _Button1_Pressed(self, channel):
        """- Chức năng: Khi bấm nút thứ nhất sẽ thực hiện các câu lệnh này 
            -Biến:None
            -Trả về:None
        """
        print("you are pressing button 1")

    def _Button2_Pressed(self, channel):
        """- Chức năng: Khi bấm nút thứ hai sẽ thực hiện các câu lệnh này 
            -Biến:None
            -Trả về:None
        """
        print("you are pressing button 2")

    def _Button3_Pressed(self, channel):
        """- Chức năng: Khi bấm nút thứ ba sẽ thực hiện các câu lệnh này 
            -Biến:None
            -Trả về:None
        """
        print("you are pressing button 3")

    def __OLED_Display(self):
        """- Chức năng: Cho phép hiển thị(Cập nhật) ra màn hình OLED(Đồng thời cập nhật vận tốc, góc lái)\n
        - Biến: None\n
        - Trả về: None
        """
        with canvas(self.__device) as draw:
            self.__Update_Angle(draw)
            self.__Printt(draw)
    def __Update_Angle(self,draw):       
        padding = 2
        top = 0
        # Move left to right keeping track of the current x position for drawing shapes.
        x = padding
        VT1 = 'VT:' + str(self.getSpeed_cm()) +'cm/s--'
        VT2 = str(self.getSpeed_rad()) +'rad/s'
        size1 = draw.textsize(VT1)
        draw.text((x, top), 'IP:'+ self.getIP(), fill=255)
        draw.text((x, top+8), VT1 ,fill=255)
        draw.text((x+size1[0], top+8), VT2,fill=255)
        draw.text((x, top+16), 'Angle:' + str(self.getAngle()),fill=255)
            
        # Draw a black filled box to clear the image. 
        draw.rectangle(self.__device.bounding_box, outline="white")

    def __Printt(self, draw):
        for i in range (0 , 4):
            if self.__is_Print[i] == 1:
                top = self.__OLED_line[i] * 8
                x = self.__OLED_left[i]
                size_text = draw.textsize(str(self.__OLED_text[i]))
                #nếu text quá dài warning
                if size_text[0] > self.__device.width:
                    print('Warning: length of text(line '+ str(self.__OLED_line[i]) + ") is out of device's width !!!")       
                draw.text((x, top), str(self.__OLED_text[i]), fill=255)
                # self.__is_Print[i] = -1
    
    
    def OLED_Print(seft, text, line = 0,left = 0):
        """- Chức năng: In ra màn hình một nội dung\n
        - Biến: \n
                + text(string): Chuỗi cần được in ra màn hình \n
                + line(int)   : Dòng cần in từ 0->4(vì 3 dòng đầu đã có hiển thị mặc định) \n
                + left(int)   : Giá trị căng trái cho con trỏ in từ 0->126: \n
        - Trả về: None
        """
        line = line + 3
        left = left + 2
        if line > 7 or line < 3 :
            print("line out of range 0 to 4")
        if left < 2 or left >128:
            # print("Warning: Left Cursor is Out of device width!!!")
            print("Left Cursor is Out of device width!!! - Text can't be seen clearly")
        index = line - 3
        
        seft.__OLED_text[index] = str(text)
        seft.__OLED_line[index] = line
        seft.__OLED_left[index] = left
        seft.__is_Print [index] = 1
    def __OLED_Logo(self):
        img_path = str(Path(__file__).resolve().parent.joinpath('images', 'CEEC_Logo_1.png'))
        logo = Image.open(img_path).convert("RGBA")
        fff = Image.new(logo.mode, logo.size, (255,) * 4)

        background = Image.new("RGBA", self.__device.size, "white")
        posn = ((self.__device.width - logo.width) // 2, 0)

        for angle in range(0, 10, 3):
            rot = logo.rotate(angle, resample=Image.BILINEAR)
            img = Image.composite(rot, fff, rot)
            background.paste(img, posn)
            self.__device.display(background.convert(self.__device.mode))
        for angle in range(10, 0, -3):
            rot = logo.rotate(angle, resample=Image.BILINEAR)
            img = Image.composite(rot, fff, rot)
            background.paste(img, posn)
            self.__device.display(background.convert(self.__device.mode))

    def OLED_Clear(self):
        """- Chức năng: Xóa toàn bộ màn hình OLED\n
        - Biến: None\n
        - Trả về: None\n
        """
        for i in range (0 , 4):
            self.__is_Print[i] = -1
        self.__device.clear()

    def setAngle(self,CarAngle  ):
        """- Chức năng: Điều chỉnh góc bẻ lái\n
        - Biến:\n
            + CarAngle(int): Giá trị góc bẻ lái từ -60 -> 60 \n
        - Trả về: None\n
        """
        if CarAngle > Car_MaxAngle or CarAngle < -Car_MaxAngle: 
            print("Angle out of range -60 to 60")
        
        CarAngle = np.clip(CarAngle,-Car_MaxAngle,Car_MaxAngle )    
        
        CarAngle = CarAngle + 90
        self.__kit.servo[ServoChannel].angle=CarAngle

    def getAngle(self):
        """- Chức năng: Lấy góc bẻ lái hiện tại\n
        - Biến: none \n
        - Trả về: Giá trị góc bẻ lái hiện tại\n
        """
        Ang = self.__kit.servo[ServoChannel].angle - 90
        return int(Ang)+1 if int(Ang) != 0 else 0
    def setSpeed_rad(self, CarSpeed: float):
        """- Chức năng: Điều chỉnh vận tốc của động cơ theo đơn vị rad/s\n
        - Biến:\n
            + CarSpeed(float): Giá trị vận tốc của động cơ từ -20 -> 20 \n
        - Trả về: None\n
        """
        if self.__MotorMode == 0:
            if abs(CarSpeed) > Car_MaxSpeed_rad:
                print("CarSpeed(rad/s) out of range -20 to 20")
            CarSpeed = np.clip(CarSpeed, -Car_MaxSpeed_rad, Car_MaxSpeed_rad)
            CarSpeed = np.round(CarSpeed,2)
            ID = 1
            inf = ""
            Buff = bytearray("N%i v%i \n"%(ID, CarSpeed),"ascii")  
            self.__serial_port.write(Buff)
            pass
        else:
            raise ValueError("Car is in Position mode. Change mode to use.")
    def setSpeed_cm(self, CarSpeed: float):
        """- Chức năng: Điều chỉnh vận tốc của động cơ theo đơn vị cm/s\n
        - Biến:\n
            + CarSpeed(float): Giá trị vận tốc của động cơ từ -58 -> 58 \n
        - Trả về: None\n
        """
        if self.__MotorMode == 0:
            if abs(CarSpeed) > Car_MaxSpeed_cm:
                print("CarSpeed(cm/s) out of range -58 to 58")
            CarSpeed = np.clip(CarSpeed, -Car_MaxSpeed_cm, Car_MaxSpeed_cm)
            CarSpeed_rad = (CarSpeed*2*3.14)/18.212

            self.setSpeed_rad(CarSpeed_rad)
        else: 
            raise ValueError("Car is in Position mode. Change mode to use.")
    def setMotorMode(self, mode):
        """- Chức năng: Thay đổi chế độ điều khiển động cơ\n
        - Biến:\n
            + mode 0 điều khiển tốc độ
            + mode 1 điều khiển vị trí
        """
        if mode == 0:
            self.__MotorMode = 0
            unl = bytearray("N1 $004=3 \n","ascii")
            self.__serial_port.write(unl)
        elif mode == 1:
            self.__MotorMode = 1
            unl = bytearray("N1 $004=2 \n","ascii")
            self.__serial_port.write(unl)
        else:
            raise ValueError("mode khong hop le")
        print("Set mode ", self.__MotorMode)
    
    def SetPosition_rad(self, position, velo):
        if self.__MotorMode == 1: 
            td = bytearray("N1 P%i v%i \n"%(position, velo),"ascii")
            self.__serial_port.write(td)
        else:
            raise ValueError("Car is in Speed mode. Change mode to use.")
        
    def SetPosition_cm(self, position, velo):
        if self.__MotorMode == 1:
            pos_rad = (position*2*3.14)/18.212
            velo_rad = (velo*2*3.14)/18.212
            td = bytearray("N1 P%i v%i \n"%(pos_rad, velo_rad),"ascii")
            print("Buffer: ",td)
            self.__serial_port.write(td)
        else:
            raise ValueError("Car is in Speed mode. Change mode to use.")
    
    def thr_motor_recv(self):
        
        while True:
            #self.__motorReqData()
            #self.__OLED_Display()
            time.sleep(0.2)
            
    def __motorUnlock(self):
        unl = bytearray("N1 O U C r \n","ascii")
        self.__serial_port.write(unl)
    
    def __motorNoACK(self):
        buff = bytearray("N1 O K0 \n","ascii")
        self.__serial_port.write(buff)

    def __motorChangeControl(self):
        buff = bytearray("N1 $005=1 \n","ascii")
        self.__serial_port.write(buff)

    def __motorDisableProtect(self):
        buff = bytearray("N1 $008=0 \n","ascii")
        self.__serial_port.write(buff)

    def __motorSaveRestart(self):
        buff = bytearray("N1 $101=1 \n","ascii")
        self.__serial_port.write(buff)

    def __motorReqData(self):
            Lenh = bytearray("N1 O G1 \n","ascii")
            self.__serial_port.write(Lenh)
            inf = self.__serial_port.read_until("\r")
            # print("inf ", inf)
            if(inf != None):
                try:
                    DataSplit = inf.split()
                    # Van toc rad/s
                    self.__vel = round(float(DataSplit[2][1:]))
                    #Dong mA
                    self.__dong = round(float(DataSplit[3][1:]))
                    #err 
                    self.__err = DataSplit[4][0:]
                except:
                    self.Motor_ClearErr()
            # Tinh van toc m/s  
            # Bán kính bánh => Chu vi = 2 * bán kính * 3.14
            # Một vòng thì sẽ đi hết ... <Chu vi bánh> => vt*chu vi bánh

    def getSpeed_rad(self) -> float:
        """- Chức năng: Lấy giá trị vận tốc hiện tại của động cơ theo đơn vị rad/s\n
        - Biến:     None \n
        - Trả về:   Giá trị vận tốc hiện tại\n
        """
        return self.__vel
    def getSpeed_round(self) -> float:
        """- Chức năng: Lấy giá trị vận tốc hiện tại của động cơ theo đơn vị vòng/s\n
        - Biến:     None \n
        - Trả về:   Giá trị vận tốc hiện tại\n
        """
        return self.__vel/(2*3.14)
    
    def getSpeed_cm(self) -> float:
        """- Chức năng: Lấy giá trị vận tốc hiện tại của động cơ theo đơn vị cm/s\n
        - Biến:     None \n
        - Trả về:   Giá trị vận tốc hiện tại\n
        """
        return round(self.getSpeed_round()*18.212, 2)

    def getMotor_Current(self) ->float:
        """
        Hàm lấy dòng tiêu thụ của động cơ
        """
        return self.__dong

    def getMotor_Err(self):
        """
        Hàm lấy lỗi của động cơ
        """
        return self.__err

    def Motor_ClearErr(self):
        cle = bytearray("N1 O C \n","ascii")
        self.__serial_port.write(cle)


    def getIP(self) -> string:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    def regBTN(self, BTNID, Function):
        GPIO.remove_event_detect(self.__BTN[BTNID-1])
        GPIO.add_event_detect(self.__BTN[BTNID-1], GPIO.RISING, Function)

    