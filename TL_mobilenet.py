#import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#os.environ["KERAS_BACKEND"] = "tensorflow"
## run python like this: export KERAS_BACKEND="plaidml.keras.backend"; python TL_mobilenet.py
## Reference https://towardsdatascience.com/celeba-attribute-prediction-and-clustering-with-keras-3d148063098d

import time
import numpy as np
import keras 
from keras import backend as K 
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam 
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import imagenet_utils, MobileNet
from keras.applications.mobilenet import preprocess_input
from myUtils import find_next_file_history, save_history, show_history, save_elapsedTime

# Some variables
imageShape=(224,224)
#imageShape=(218,178) #Celeba croped image shape
histFileName = 'historyMobilenet.csv'
dirHistFileName = './history'
numEpochs=20

# Applying TranserLearning, we freeze the base layer and retrain the one o nthe top
start = time.time()
starting_model = MobileNet(input_shape=imageShape+(3,), alpha = 0.75,depth_multiplier = 1,
                           dropout = 0.001,include_top = False, weights = "imagenet", classes = 1000,
                           backend=keras.backend, layers=keras.layers,models=keras.models,utils=keras.utils) # this line imports the mobilenet model trained on imagenet dataset and discard the last 1000 neurons layer 

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

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# load and iterate training dataset
train_generator = train_datagen.flow_from_directory('./orient/train',
                                                    target_size=imageShape,
                                                    color_mode='rgb',
                                                    batch_size = 32,
                                                    class_mode = 'categorical',
                                                    shuffle= True)

val_generator = train_datagen.flow_from_directory('./orient/valid',
    target_size=imageShape,
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

# Lets re-traing the top layers, this step may require some time depending on yor PC/GPU 
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
step_size_train = train_generator.n//train_generator.batch_size
step_size_val = val_generator.n//val_generator.batch_size
history = model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train,epochs=numEpochs,
                    validation_data=val_generator,validation_steps=step_size_val)

model.save('./models/new_celeba_model.h5')
end = time.time()
elapsedTime= (end - start)
print("Elapsed Time:")
print("\t\t{:.3f}s".format(elapsedTime))

show_history(history)
finalHistoryFile=find_next_file_history(dirHistFileName,histFileName)
save_history(history.history,finalHistoryFile)

save_elapsedTime(elapsedTime,finalHistoryFile)




