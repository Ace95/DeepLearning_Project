import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import keras 
from keras import backend as K 
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.mobilenet import decode_predictions
import cv2
import PIL.Image as Image
from keras.applications.vgg19 import preprocess_input

# Some variables
imageShape=(224,224)
#imageShape=(218,178) #Celeba croped image shape

imageFile='./CENTER.jpg'
#imageFile='./LEFT.jpg'
#imageFile='./RIGHT.jpg'


# Image pre-processing (MobileNet accepts 224x224 images as input)
labels = ['left', 'right','center']


def prepare_image(file):

#    img_path = './images/'
    img_path = ''
    img = image.load_img(img_path + file, target_size=imageShape)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

#    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    return keras.applications.vgg19.preprocess_input(img_array_expanded_dims)

model = load_model('./models/VGG19_celeba_model.h5')
preprocessed_image = prepare_image(imageFile)
predictions = model.predict(preprocessed_image)


#results = decode_predictions(predictions)
#print(results)

# myTest = Image.open(imageFile)
# myTest = np.array(myTest)/255.0
# print(myTest.shape)

imgCeleba = cv2.imread(imageFile)
cv2.imshow('Celeba Image',imgCeleba)




print('+---------------+------------+')
print('| {:^6} \t\t| {:^10} |'.format('LABEL','Accuracy'))
print('+---------------+------------+')
for i in range(len(labels)):
    print('| {:6} \t\t| \t {:05.2f} % |'.format(labels[i].upper(),predictions[0][i]*100))
print('+---------------+------------+')
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

    