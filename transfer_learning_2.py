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
data_dir = '/home/naruto/PycharmProjects/data/'
output_dir = "output"
ckpt_dir = "ckpt_dir"

FLAG = None


#assert
# preprocessing_function=preprocess_input
height=224

train_data='/home/naruto/PycharmProjects/knifey_spoony_demo/data/knifey-spoony/train_data_link/'
test_data='/home/naruto/PycharmProjects/knifey_spoony_demo/data/knifey-spoony/test/'
gen = ImageDataGenerator()
train_data = gen.flow_from_directory(train_data, target_size=(height, height),batch_size=64)
generator_test = gen.flow_from_directory(test_data, target_size=(height, height),batch_size=64)
base_model = ResNet50(input_tensor=Input((height, height, 3)),weights='imagenet', include_top=False)
base_model.summary()
for layer in base_model.layers:
    layer.trainable = False

# transfer_layer = base_model.get_layer('activation_48')
# new_model=Model(inputs=base_model.input,outputs=base_model.get_layer('activation_48').output)
# model1=Sequential()
# model1.add(Flatten(input_shape=(new_model.output.shape[1:])))
# model1.add(Dense(512,activation='sigmoid'))
# model1.add(Dense(64,activation='sigmoid'))
# model1.add(Dense(3,activation='sigmoid'))


x=Flatten()(base_model.layers[172].output)
# x=Dense(512,activation='sigmoid')(x)
# x=Dense(64,activation='sigmoid')(x)
x=Dense(3,activation='sigmoid')(x)
#
model1 = Model(inputs=base_model.input, outputs=x)
model1.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_ft = model1.fit_generator(
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

