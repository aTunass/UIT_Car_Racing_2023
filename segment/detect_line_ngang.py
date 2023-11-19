import cv2
import os
import numpy as np
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
def remove_small_contours(image):
    image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
    image_remove = cv2.bitwise_and(image, image, mask=mask)
    return image_remove
def find_line(image):
    image = image[200:]
    image = cv2.resize(image, (160, 80))
    # Chuyển đổi ảnh sang không gian màu HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tạo mặt nạ cho vùng màu trắng
    lower_white = np.array([0, 0, 155], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Áp dụng mặt nạ để giữ màu trắng và chuyển phần còn lại thành màu đen
    result = cv2.bitwise_and(image, image, mask=mask_white)
    # Chuyển ảnh thành ảnh nhị phân
    _, binary_result = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)
    try:
        binary_result = remove_small_contours(binary_result)
    except:
        binary_result = binary_result
    return binary_result
def detect_hor(img):
    return line_ver(img, 40) + line_ver(img, 60) + line_ver(img, 80) + line_ver(img, 100) + line_ver(img, 120)
# for img_name in os.listdir("segment"):
#     print(os.path.join("segment",img_name))
#     img = cv2.imread(os.path.join("segment",img_name))
#     binary_result = find_line(img)
#     if detect_hor(binary_result)>3:
#         cv2.imwrite(f"output_hor_detect/{img_name}.jpg", img)
print(np.zeros((640,640,3)))