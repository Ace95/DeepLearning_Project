# DeepLearning_Project
Final project for the Deep Learning Course at UNIBO

## Treating data - selectImage.py
This program helps to generate the training data set.
It expects a directory "./img_align_celeba' with cropped images from celeba.
Using the keys a, k and spacebar or "left arrow", "right arrow" and "up/down arrow" classify the face orientation as left, right or center, respectively. 

After the pose is selected, files are moved to the directory "./orient/" in a subfolder depending on the user classification (center, left and right).

# Running the Google Colab
Access google colab: https://colab.research.google.com/notebooks/intro.ipynb#recent=true
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
