#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:42:24 2020

@author: Nicolaas
Inspired on:
https://github.com/khanhnamle1994/fashion-mnist/blob/master/VGG19-GPU.ipynb

"""
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import time
import keras 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.preprocessing.image  import ImageDataGenerator
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input

from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras import models
from keras import layers
from keras import optimizers
from keras.applications.vgg19 import preprocess_input



trainDIR='./orient/train'
valDIR='./orient/valid'
imgHeight=224 #218
imgWidth=224 #178
imageShape=(imgWidth,imgHeight) #Celeba croped image shape


# https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

# load and iterate validation dataset
#val_it = datagen.flow_from_directory('data/validation/', class_mode='categorical', batch_size=64)
# # load and iterate test dataset
# #test_it = datagen.flow_from_directory('data/test/', class_mode='categorical', batch_size=64)
# test_it = datagen.flow_from_directory(
#     directory=valDIR,
#     target_size=(224, 224),
#     color_mode="rgb",
#     batch_size=32,
#     class_mode="categorical",
#     shuffle=False
# )

# Create the base model of VGG19
#vgg19 = VGG19(weights='imagenet', include_top=False, input_shape = (150, 150, 3), classes = 10)

# Applying TranserLearning, we freeze the base layer and retrain the one o nthe top
start = time.time()

starting_model = VGG19(input_shape=imageShape+(3,), include_top = False, weights = "imagenet", classes = 1000,
                            backend=keras.backend, layers=keras.layers,models=keras.models,utils=keras.utils) # this line imports the VGG19 model trained on imagenet dataset and discard the last 1000 neurons layer 

x = starting_model.output 
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
#train_it = datagen.flow_from_directory('data/train/', class_mode='categorical', batch_size=64)

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

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
step_size_train = train_it.n//train_it.batch_size
history = model.fit_generator(generator=train_it,steps_per_epoch=step_size_train,
                    epochs=10,validation_data=val_it)


model.save('./models/VGG19_celeba_model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

print (acc,val_acc)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()

# score = model.evaluate_generator(val_it,train_it.n//train_it.batch_size, verbose=1)
# # nb_validation_samples/batch_size, workers=12)
# scores = model.predict_generator(val_it,train_it.n//train_it.batch_size, verbose=1)

# correct = 0
# for i, n in enumerate(val_it.filenames):
#     print (i,n)
#     print(scores)
#     if n.startswith("left") and scores[i][0] <= 0.5:
#         correct += 1
#     if n.startswith("right") and scores[i][0] > 0.5:
#         correct += 1
#     if n.startswith("center") and scores[i][0] > 0.5:
#         correct += 1

# print("Correct:", correct, " Total: ", len(val_it.filenames))
# print("Loss: ", score[0], "Accuracy: ", score[1])
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

end = time.time()
elapsedTime= (end - start)
print("Elapsed Time:")
print("\t\t{:.2f}m".format(elapsedTime/60))

