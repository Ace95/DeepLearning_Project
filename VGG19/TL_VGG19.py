#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:42:24 2020

@author: Nicolaas
Inspired on:
https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a

"""
#import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
## up or run python like this: export KERAS_BACKEND="plaidml.keras.backend"; python TL_VGG19.py
import time
import keras
import pandas as pd
import matplotlib.pyplot as plt
from numpy import asarray
from keras.utils import to_categorical
from keras.preprocessing.image  import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
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
# imgHeight=224 #218
# imgWidth=224 #178
imgHeight=218
imgWidth=178
imageShape=(imgHeight,imgWidth) #Celeba croped image shape

histFileName = 'historyVGG19.csv'
dirHistFileName = './history'

numEpochs=20


# https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

# Applying TransferLearning, we freeze the base layer and retrain the one o nthe top
start = time.time()

vgg = VGG19(input_shape=imageShape+(3,), include_top = False, weights = "imagenet", classes = 1000,
                            backend=keras.backend, layers=keras.layers,
                            models=keras.models,utils=keras.utils) # this line imports the VGG19 model trained on imagenet dataset and discard the last 1000 neurons layer 

output=vgg.layers[-1].output
output = keras.layers.Flatten()(output)

vgg_model = Model(vgg.input,output)

# We want to use the pre-trained weights
vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False
        
##
input_shape = vgg_model.output_shape[1]

model = keras.models.Sequential()
model.add(vgg_model)
model.add(keras.layers.InputLayer(input_shape=(input_shape,)))
model.add(Dense(512,activation='relu',input_dim=input_shape))
model.add(Dropout(0.7))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(3,activation='softmax'))


##

# layers = [(layer.name, layer.trainable) for layer in model.layers]
# print(pd.DataFrame(layers, columns=['Layer Name', 'Layer Trainable']))

# Load training data and test data with Imagenerator for on demand loading files
# create a data generator

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# load and iterate training dataset
train_it = train_datagen.flow_from_directory(
    directory=trainDIR,
    target_size=imageShape,
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

val_it = train_datagen.flow_from_directory(
    directory=valDIR,
    target_size=imageShape,
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

# # Show some of our images
# examples = [next(train_it) for i in range(0,5)]
# fig, ax = plt.subplots(1,5, figsize=(32, 12))
# print('Labels:', [item[1][0] for item in examples])

# [ print(examples[i][0][0].shape) for i in range(0,4)]

# l = [ax[i].imshow(examples[i][0][0]) for i in range(0,4)]
# image_file = plt.imread(trainDIR+'/center_pose/000005.jpg')
# ax[4].imshow(image_file)
# plt.show()
# exit(1)

# Lets re-traing the top layers, this step may require some time depending on yor PC/GPU 

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['acc'])
step_size_train = train_it.n//train_it.batch_size
step_size_val = val_it.n//val_it.batch_size

history = model.fit_generator(generator=train_it,steps_per_epoch=step_size_train,
                    epochs=numEpochs,
                    validation_data=val_it,
                    validation_steps=step_size_val
                    )


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


