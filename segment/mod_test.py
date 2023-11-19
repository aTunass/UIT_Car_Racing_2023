import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model, load_model
from keras.src.layers.merging.concatenate import concatenate
from keras import regularizers
from keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


import os
import cv2
from glob import glob
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def create_model(input_shape, pool_size):
    # Create the actual neural network here
    input_img = Input(input_shape, name='img')

    c1_1 = Conv2D(8, (3, 3), activation='relu', padding='same') (input_img)
    c1_2 = Conv2D(8, (3, 3), activation='relu', padding='same') (input_img)
    c1 = concatenate([c1_1, c1_2])
    p1 = MaxPooling2D((2, 2)) (c1)

    c2_1 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2_2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = concatenate([c2_1, c2_2])
    p2 = MaxPooling2D((2, 2)) (c2)

    c3_1 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3_2 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = concatenate([c3_1, c3_2])
    p3 = MaxPooling2D((2, 2)) (c3)

    u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (p3)
    u5 = concatenate([u5, c3])
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same') (u5)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c2])
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c1])
    c8 = Conv2D(8, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(8, (3, 3), activation='relu', padding='same') (c8)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

    model = Model(inputs=[input_img], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
    return model



pool_size = (2, 2)
input_shape = (40, 160, 1)

model = create_model(input_shape, pool_size)                #Xây dựng model

#bắt đầu training
model = load_model('/home/tuan/Desktop/UIT_Car_Racing_2022/weight_seg_new/model-093_12.h5')
for img_name in os.listdir("segment"):
    print(os.path.join("segment",img_name))
    img = cv2.imread(os.path.join("segment",img_name), 0)
    img = img[100:, :]
    img = np.expand_dims(img, 2)
    img = cv2.resize(img, (img.shape[1]//4, 40))
# cv2.imwrite("pre_img.png", img)


# mask = cv2.imread("label/image_origin_2 (2).png")

# mask = mask[200:, :, :]

# mask= cv2.resize(mask, (mask.shape[1]//4, mask.shape[0]//4))
# # mask = mask.reshape(mask.shape[0], mask.shape[1] ,1)
# cv2.imwrite("pre_mask.png", mask)

    img = np.expand_dims(img, 2)
    img = img/255
    img = np.array(img, dtype=np.float32)
    print(img.shape)
    img = img[None, :, :, :]
    pred = model.predict(img, 1)
    pred = np.squeeze(pred)
    pred = np.where(pred > 0.5, 255, 0)
    pred = pred.astype(np.uint8)
    cv2.imwrite(os.path.join("test",img_name), pred)