import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model, load_model
from keras.layers.merge import concatenate
from keras import regularizers
from keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau

import os
import cv2
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt

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
    model.summary()
    return model



filenames = sorted(glob("img/*"))
labelf= sorted(glob("label/*"))   

filenames.sort()
labelf.sort()
labels=[]
images=[]

def augment_hsv(im, hgain= 0, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed


#đọc tất cả ảnh
for path in filenames:
    img = cv2.imread(path, 0)
    # augment_hsv(img) if np.random.randint(2) == 1 else 0
    # img = img[200:, :, :]
    img = img[200:, :]
    img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
    img = img/255
    img = np.expand_dims(img, 2)
    images.append(img)
#đọc tất cả label
for path in labelf:
    mask = plt.imread(path)
    if len(mask.shape) == 3:
      mask = mask[200:, :, 0]
      mask = np.where(mask != 0, 1.0, 0.0)
    else:
      mask = mask[200:, :]

    mask= cv2.resize(mask, (mask.shape[1]//4, mask.shape[0]//4))
    # print(mask.shape)
    mask=mask.reshape(mask.shape[0], mask.shape[1] ,1)
    labels.append(mask)

print(np.array(labels).shape)

images = np.array(images)
labels = np.array(labels)
images, labels = shuffle(images, labels)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2)

batch_size = 128    #batch size
epochs = 100     #số epochs
pool_size = (2, 2)
input_shape = X_train.shape[1:]
print(input_shape)
model = create_model(input_shape, pool_size)                #Xây dựng model

earlystop = EarlyStopping(patience=10)   
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',   #điều cần quan tâm là độ chính xác nếu độ chính xác 
                                                                 #không tăng thì lr sẽ giảm
                                            patience=3,          #số lượng epoch mà sau đó không cải thiện chương trình sẽ stop
                                            verbose=1,           #có cập nhật và hiển thị trạng thái khi reduce
                                            factor=0.5,          #new_lr=factor*lr
                                            min_lr=1e-6)      #giới hạn lr
checkpoint = ModelCheckpoint('weight_mod_gray/model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')
callb=[learning_rate_reduction,checkpoint]

#bắt đầu training
# model = load_model('weight_ute/pretrained.h5')
history = model.fit(X_train, y_train ,  validation_data=(X_val, y_val),
                    epochs=epochs, verbose=1, callbacks=callb)