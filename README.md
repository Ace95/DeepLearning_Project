# DeepLearning_Project
Final project for the Deep Learning Course at UNIBO

## Treating data - selectImage.py
This program helps to generate the training data set.
It expects a directory "./img_align_celeba' with cropped images from celeba.
Using the keys a, k and spacebar or "left arrow", "right arrow" and "up/down arrow" classify the face orientation as left, right or center, respectively. 

After the pose is selected, files are moved to the directory "./orient/" in a subfolder depending on the user classification (center, left and right).

## Running on Google Colab
Access google colab: https://colab.research.google.com/notebooks/intro.ipynb#recent=true

Set the GPU on the colab machine: Menu "Runtime"--> "Change Runtime type", select GPU.

Create a new Notebook: "cloneDLProject"

On the first cell, copy this code:
```
from getpass import getpass
import os
user = getpass('User')
password = getpass('Password')
os.environ['GIT_AUTH'] = user + ':' + password
!git clone https://$GIT_AUTH@github.com/Ace95/DeepLearning_Project.git
```

On another cell, to run a python program, got to the project directory and run like the following example:
```
%cd /content/DeepLearning_Project/CNNfromScratch
!python CNN4.py
```
On another cell, plot the results from the generated csv file
```
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')
df = pd.read_csv('./history/historyCNN4.csv')
plt.plot(df.val_loss,'red',label='Training Loss')
plt.plot(df.loss,'blue',label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Loss - Validation and Trainng')
plt.legend()
plt.show()

plt.style.use('ggplot')
plt.plot(df.val_acc,'red',label='Training Accuracy')
plt.plot(df.acc,'blue',label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Accuracy - Validation and Trainng')
plt.legend()
plt.show()
```
If everthing goes wrong we can always restart! Menu "Runtime" --> "Factory reset runtime". And then Reconnect.
