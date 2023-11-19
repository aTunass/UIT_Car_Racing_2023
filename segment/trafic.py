import cv2
import os
import numpy as np
import pybboxes as pbx
def traffic_light_det(img, box): # boxes: x,y,w,h
        '''
        input: img, box
        output: 0: red, 1: yellow, 2: green
        '''
        # self.GHSVLOW = np.array([45, 100, 100])
        # self.GHSVHIGH = np.array([90, 255, 255])
        # self.YHSVLOW = np.array([20, 100, 100])
        # self.YHSVHIGH = np.array([40, 255, 255])
        # self.RHSVLOW = np.array([160,100,100])
        # self.RHSVHIGH = np.array([180,255,255])
        # self.RHSVLOW_1 = np.array([0,70,50])
        # self.RHSVHIGH_1 = np.array([10,255,255])
        GHSVLOW = np.array([45, 100, 100])
        GHSVHIGH = np.array([90, 255, 255])
        YHSVLOW = np.array([20, 100, 100])
        YHSVHIGH = np.array([40, 255, 255])
        RHSVLOW = np.array([160,100,100])
        RHSVHIGH = np.array([180,255,255])
        RHSVLOW_1 = np.array([0,70,50])
        RHSVHIGH_1 = np.array([10,255,255])
        # TODO: Check boxes
        x1, y1 = box[:2]
        x2, y2 = box[2:]
        img_crop = img[int(y1):int(y2), int(x1):int(x2)]
        img_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        # maskg = cv2.inRange(img_hsv, self.GHSVLOW, self.GHSVHIGH)
        # masky = cv2.inRange(img_hsv, self.YHSVLOW, self.YHSVHIGH)
        # maskr_1 = cv2.inRange(img_hsv, self.RHSVLOW, self.RHSVHIGH)
        # maskr_2 = cv2.inRange(img_hsv, self.RHSVLOW_1, self.RHSVHIGH_1)
        maskg = cv2.inRange(img_hsv, GHSVLOW, GHSVHIGH)
        masky = cv2.inRange(img_hsv, YHSVLOW, YHSVHIGH)
        maskr_1 = cv2.inRange(img_hsv, RHSVLOW, RHSVHIGH)
        maskr_2 = cv2.inRange(img_hsv, RHSVLOW_1, RHSVHIGH_1)
        maskr = maskr_1 | maskr_2

        area = [check(mask) for mask in [maskr, masky, maskg]]
        index = area.index(max(area))
        return index
def check(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return 0
    max_contour = max(contours, key = cv2.contourArea)
    area = cv2.contourArea(max_contour)
    return area
# W, H = 640, 640  # WxH of the image
# for img_name in sorted(os.listdir("trafic_light")):
#     print(os.path.join("trafic_light",img_name))
#     with open(os.path.join("labels",img_name[:-4]+".txt"), 'r') as file:
#         line = file.readline()
#         values = line.split()
#         values = [float(val) for val in values]
#     img = cv2.imread(os.path.join("trafic_light",img_name))
#     img = cv2.resize(img, (640, 640))
#     box = pbx.convert_bbox(values[-4:], from_type="yolo", to_type="voc", image_size=(W, H))
#     output = traffic_light_det(img, box)
#     print(output)

import cv2

# Tạo đối tượng VideoCapture để mở camera. Thường là camera mặc định (0), nhưng bạn cũng có thể thử các giá trị khác như 1, 2, ...
cap = cv2.VideoCapture(0)
import time
# Kiểm tra xem camera có được mở thành công không
if not cap.isOpened():
    print("Không thể mở camera. Đảm bảo rằng camera được kết nối và không được sử dụng bởi ứng dụng khác.")
else:
    while True:
        # Đọc frame từ camera
        ret, frame = cap.read()

        # Hiển thị frame
        cv2.imshow('Camera', frame)

        # Kiểm tra xem người dùng có nhấn 'ESC' để thoát không
        key = cv2.waitKey(1) & 0xFF  # Đọc mã ASCII của phím

        if key == ord('a'):
            print("Nút 'a' đã được nhấn.")
            t = time.time()
            # while(time.time()-t<2):
            #      None
            time.sleep(2)
        elif key == 27:  # 27 là mã ASCII của phím ESC
            break

    # Giải phóng camera và đóng cửa sổ hiển thị khi kết thúc
    cap.release()
    cv2.destroyAllWindows()
