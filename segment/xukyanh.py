import cv2 
import numpy as np
import os
# for img_name in os.listdir("test_dataOD_bysegment"):
#     print(os.path.join("test_dataOD_bysegment",img_name))
#     img = cv2.imread(os.path.join("test_dataOD_bysegment",img_name))
#     img = img[100:, :]
#     #img = np.expand_dims(img, 2)
#     img = cv2.resize(img, (img.shape[1]//4, 40))
"""r 180-80 g 170-70 b170"""
import cv2
import numpy as np
def remove_small_contours(image):
    image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
    image_remove = cv2.bitwise_and(image, image, mask=mask)
    return image_remove
# Đọc ảnh
image = cv2.imread('data_seg_2/b_240.jpg')
def fiind_line(image):
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
# Hiển thị ảnh gốc và kết quả
for img_name in os.listdir("segment"):
    print(os.path.join("segment",img_name))
    img = cv2.imread(os.path.join("segment",img_name))
    binary_result = fiind_line(img)
    cv2.imwrite(f"testunet2/{img_name}.jpg", binary_result)
# binary_result = fiind_line(image)
# cv2.imshow('Original Image', image)
# cv2.imshow('Result Image', binary_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


