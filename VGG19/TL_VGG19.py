#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:42:24 2020

@author: Nicolaas
Inspired on:
https://github.com/khanhnamle1994/fashion-mnist/blob/master/VGG19-GPU.ipynb

"""
#import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
## up or run python like this: export KERAS_BACKEND="plaidml.keras.backend"; python TL_VGG19.py
import time
import keras
from keras.utils import to_categorical
from keras.preprocessing.image  import ImageDataGenerator
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input

from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras import models
from keras import layers
from keras import optimizers
from  myUtils import find_next_file_history, save_history, show_history, save_elapsedTime



trainDIR='../orient/train'
valDIR='../orient/valid'
imgHeight=224 #218
imgWidth=224 #178
imageShape=(imgWidth,imgHeight) #Celeba croped image shape

histFileName = 'historyVGG19.csv'
dirHistFileName = './history'

numEpochs=20


# https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

# Applying TransferLearning, we freeze the base layer and retrain the one o nthe top
start = time.time()

starting_model = VGG19(input_shape=imageShape+(3,), include_top = False, weights = "imagenet", classes = 1000,
                            backend=keras.backend, layers=keras.layers,
                            models=keras.models,utils=keras.utils) # this line imports the VGG19 model trained on imagenet dataset and discard the last 1000 neurons layer 

# x = starting_model.output 
x = (starting_model.output)
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense (512,activation='relu')(x)
preds = Dense(3,activation='softmax')(x)  # Note that number of neurons in the last layer depends on the number of classes you want to detect
model = Model(inputs=starting_model.input,outputs=preds)

# We want to use the pre-trained weights

for layer in model.layers[:86]:
    layer.trainable=False
for layer in model.layers[86:]:
    layer.trainable=True

# Load training data and test data with Imagenerator for on demand loading files
# create a data generator

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# load and iterate training dataset
train_it = train_datagen.flow_from_directory(
    directory=trainDIR,
    target_size=(imgHeight, imgWidth),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

val_it = train_datagen.flow_from_directory(
    directory=valDIR,
    target_size=(imgHeight, imgWidth),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

# Lets re-traing the top layers, this step may require some time depending on yor PC/GPU 

#model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['acc'])
step_size_train = train_it.n//train_it.batch_size
step_size_val = val_it.n//val_it.batch_size
history = model.fit_generator(generator=train_it,steps_per_epoch=step_size_train,
                    epochs=numEpochs,validation_data=val_it,validation_steps=step_size_val)


model.save('./models/VGG19_celeba_model.h5')

##    
end = time.time()
elapsedTime= (end - start)
print('Elapsed Time:')
print("\t\t{:.2f}m".format(elapsedTime/60))

show_history(history)
finalHistoryFile=find_next_file_history(dirHistFileName,histFileName)
save_history(history.history,finalHistoryFile)
save_elapsedTime(elapsedTime,finalHistoryFile)


