#encoding=utf-8

import os, sys

from datetime import datetime
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import keras

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import h5py
from keras.applications.inception_v3 import preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.system("nvidia-smi | grep vpython3 | awk '{print $3}' | xargs kill -9")


height=224
train_data='/home/naruto/PycharmProjects/knifey_spoony_demo/data/knifey-spoony/train_data_link/'
test_data='/home/naruto/PycharmProjects/knifey_spoony_demo/data/knifey-spoony/test/'
gen = ImageDataGenerator()
train_data = gen.flow_from_directory(train_data, target_size=(height, height),batch_size=64)
generator_test = gen.flow_from_directory(test_data, target_size=(height, height),batch_size=64)
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
base_model = ResNet50(input_tensor=Input((height, height, 3)),weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
# x=MaxPooling2D()(x)
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(3, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:

# for layer in model.layers[:172]:
#    layer.trainable = False
# for layer in model.layers[172:]:
#    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history_ft = model.fit_generator(
    train_data,#可自定义
    # samples_per_epoch=4170,  # nb_train_samples，Basically steps_per_epoch = samples_per_epoch/batch_size
    # steps_per_epoch=10,  # nb_train_samples#每轮epoch遍历的samples
    validation_data=generator_test,#可自定义
    nb_epoch=100,
    verbose=1,#控制显示方式，冗长
    validation_steps=530//64,
    workers=8,
    use_multiprocessing=True,
    # epochs=100
    # nb_val_samples=530 # nb_val_samples`->`validation_steps
)