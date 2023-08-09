import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, UpSampling2D, concatenate, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import os
import cv2
SIZE_IMG = 128

def dice(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
model = tf.keras.models.load_model('mymodel-pretrain.h5', custom_objects = {'dice':dice})

import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure
import time
img_raw = cv2.imread('/content/drive/MyDrive/img_test/Capture5.JPG')
img_org = img_raw.copy()
img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
h,w = img.shape
img = cv2.resize(img,(128,128))
arr_img = np.array(img)/255.0
arr_img = arr_img.reshape(-1, 128,128,1)
res = model.predict(arr_img)
# plt.imshow(res[0],cmap = plt.cm.binary)
# plt.show()
cv2.imwrite('/content/drive/MyDrive/img_test/out.jpg',res[0]*255)
mask = np.array(cv2.resize(res[0]*255,(w,h)), np.uint8)
# mask[mask > 70] = 1
# mask[mask <= 70] = 0
print(mask[mask>70])

def changeValue(img, mask, max_range, min_range, axis = 2):
  value = img[:,:,axis]
  x = value[mask>50]
  if (len(x)==0):
    return img
  if (x.max() == x.min()):
    index = np.where(mask>50)
    for i,j in zip(index[0],index[1]):
        img[i,j,axis] = (max_range + min_range)/2
    return np.array(img ,np.uint8)
  else:
    index = np.where(mask>50)
    ratio = (max_range-min_range)/(x.max()-x.min())
    x = x*ratio + min_range - x.min()*ratio
    n = 0
    for i,j in zip(index[0],index[1]):
        img[i,j,axis] = x[n]
        n += 1
    return np.array(img ,np.uint8)

def changeColor(img , mask, img_org, r = 0, g = 0, b = 0):
  # color = 0
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  img = changeValue(img, mask, r, 0,2)
  img = changeValue(img, mask, g, 0,1)
  img = changeValue(img, mask, b, 0,0)
  img = np.hstack((img_org, img))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img)
  plt.show()

changeColor(img_raw, mask, img_org, 255, 51, 51)
changeColor(img_raw, mask, img_org, 255, 128, 0)
changeColor(img_raw, mask, img_org, 255,255, 0)
changeColor(img_raw, mask, img_org, 0, 255, 0)
changeColor(img_raw, mask, img_org, 0, 50, 255)
changeColor(img_raw, mask, img_org, 127, 0, 255)