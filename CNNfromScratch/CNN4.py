#import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#os.environ["KERAS_BACKEND"] = "tensorflow"
## run python like this: export KERAS_BACKEND="plaidml.keras.backend"; python TL_mobilenet.py
## Reference https://nbviewer.jupyter.org/github/khanhnamle1994/fashion-mnist/blob/master/CNN-4Conv.ipynb
import time

import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image  import ImageDataGenerator
from  myUtils import find_next_file_history, save_history, show_history, save_elapsedTime

# Some variables
#imageShape=(224,224)
imageShape=(218,178) #Celeba croped image shape
histFileName = 'historyCNN4.csv'
dirHistFileName = './history'
numEpochs=20
modelsFile = './models/CNN4_celeba_model.h5'
trainDIR='../orient/train'
valDIR='../orient/valid'


# Applying TranserLearning, we freeze the base layer and retrain the one o nthe top
start = time.time()

cnn4 = Sequential()
cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=imageShape+(3,))) #diff from example 
cnn4.add(BatchNormalization())

cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))

cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))

cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))

cnn4.add(Flatten())

cnn4.add(Dense(512, activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.7))

cnn4.add(Dense(512, activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.7))

cnn4.add(Dense(3, activation='softmax')) ## Final number of categories

cnn4.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['acc'])

train_datagen = ImageDataGenerator()

# load and iterate training dataset
train_generator = train_datagen.flow_from_directory(trainDIR,
                                                    target_size=imageShape,
                                                    color_mode='rgb',
                                                    batch_size = 32,
                                                    class_mode = 'categorical',
                                                    shuffle= True)

val_generator = train_datagen.flow_from_directory(valDIR,
    target_size=imageShape,
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

# examples = [next(train_generator) for i in range(0,5)]
# fig, ax = plt.subplots(1,5, figsize=(32, 12))
# print('Labels:', [item[1][0] for item in examples])

# [ print(examples[i][0][0].shape) for i in range(0,4)]

# l = [ax[i].imshow(examples[i][0][0]) for i in range(0,4)]
# image_file = plt.imread(trainDIR+'/center_pose/000005.jpg')
# ax[4].imshow(image_file)
# plt.show()
# exit(1)
step_size_train = train_generator.n//train_generator.batch_size
step_size_val = val_generator.n//val_generator.batch_size
history = cnn4.fit_generator(generator=train_generator,steps_per_epoch=step_size_train,epochs=numEpochs,
                    validation_data=val_generator,validation_steps=step_size_val)

cnn4.save('./models/CNN4_celeba_model.h5')
end = time.time()
elapsedTime= (end - start)
print("Elapsed Time:")
print("\t\t{:.3f}m".format(elapsedTime/60))

show_history(history)
finalHistoryFile=find_next_file_history(dirHistFileName,histFileName)
save_history(history.history,finalHistoryFile)

save_elapsedTime(elapsedTime,finalHistoryFile)




