import cv2
import numpy as np 


def check(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return 0
    max_contour = max(contours, key = cv2.contourArea)
    area = cv2.contourArea(max_contour)
    return area

'''

# '''
def traffic_light_det(self, img, box): # boxes: x,y,w,h
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
    maskg = cv2.inRange(img_hsv, self.GHSVLOW, self.GHSVHIGH)
    masky = cv2.inRange(img_hsv, self.YHSVLOW, self.YHSVHIGH)
    maskr_1 = cv2.inRange(img_hsv, self.RHSVLOW, self.RHSVHIGH)
    maskr_2 = cv2.inRange(img_hsv, self.RHSVLOW_1, self.RHSVHIGH_1)
    maskr = maskr_1 | maskr_2

    area = [self.check(mask) for mask in [maskr, masky, maskg]]
    index = area.index(max(area))
    return index
     
# def traffic_light_det(img): # boxes: x,y,w,h
#     '''
#     input: img, box
#     output: 0: red, 1: yellow, 2: green
#     '''
#     GHSVLOW = np.array([45, 100, 100])      #green
#     GHSVHIGH = np.array([90, 255, 255])  
#     YHSVLOW = np.array([20, 100, 100])      #yellow
#     YHSVHIGH = np.array([40, 255, 255])     
#     RHSVLOW = np.array([160,100,100])       #red 
#     RHSVHIGH = np.array([180,255,255])      
#     RHSVLOW_1 = np.array([0,70,50])         #red2
#     RHSVHIGH_1 = np.array([10,255,255])
#     # TODO: Check boxes
#     # x1, y1 = box[:2]
#     # x2, y2 = box[2:]
#     # img_crop = img[int(y1):int(y2), int(x1):int(x2)]

#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
#     mask_green = cv2.inRange(img_hsv, GHSVLOW, GHSVHIGH)
#     mask_yellow = cv2.inRange(img_hsv, YHSVLOW, YHSVHIGH)
#     mask_red_1 = cv2.inRange(img_hsv, RHSVLOW, RHSVHIGH)
#     mask_red_2 = cv2.inRange(img_hsv, RHSVLOW_1, RHSVHIGH_1)
#     mask_red = mask_red_1 | mask_red_2

#     area = [check(mask) for mask in [mask_red, mask_yellow, mask_green]]
#     index = area.index(max(area))
#     return index


path = "traffic/a_7360.jpg"

## Read
img = cv2.imread(path)
cv2.imshow("img", img)

a = traffic_light_det(img)
print("index = ", a)

cv2.waitKey(0)
cv2.destroyAllWindows() 
