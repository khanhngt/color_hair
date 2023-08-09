import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, UpSampling2D, concatenate, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import os
from model import unet_model
import time

SIZE_IMG = 128
DATA_DIR = '/content/drive/MyDrive/DeepLearning/Dataset/improvedataset/'
lib = ['val', 'train_mask','train','val_mask']
data = {}

#load data from pickle file
import pickle
for name in lib:
  pickle_in = open(os.path.join(DATA_DIR, 'data_{}.pickle'.format(name)),'rb')
  data[name] = pickle.load(pickle_in)
  pickle_in.close()

data['train'] = data['train'][:8000]
data['val'] = data['val'][:2000]
data['train_mask'] = data['train_mask'][:8000]
data['val_mask'] = data['val_mask'][:2000]
for name in lib:
  print(data[name].shape)
  print(name)

#config
input_size = (SIZE_IMG,SIZE_IMG,1)
base = 5
model = unet_model(base, input_size)



NAME = "hair-unet-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

earlyStopping = EarlyStopping(monitor='loss', patience=3)

checkpoint = ModelCheckpoint(os.path.join(DATA_DIR, 'mymodel.h5'), monitor='dice', verbose=1,
                             mode='max',
                             save_best_only=True, save_weights_only=False, period=1, )

model.fit(data['train'], data['train_mask'], epochs = 100, batch_size=32, validation_data = (data['val'], data['val_mask']), callbacks = [tensorboard, earlyStopping, checkpoint])
model.save(os.path.join(DATA_DIR,'mymodel.h5'))
