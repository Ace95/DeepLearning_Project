#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:40:21 2020

@author: Nicolaas Ruberg
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

def save_history(history,fileName):
    hist_df = pd.DataFrame(history)
    with open(fileName,mode='w') as f:
        hist_df.to_csv(f)

def save_elapsedTime(elapsedTime,fileName):
    f = open(fileName,'a+')
    f.write('%%Elapsed Time: {:.2f}m. \n'.format(elapsedTime/60))
    f.close()
        
def find_next_file_history(dirHistory, fileName):
    if not os.path.exists(dirHistory):
        os.makedirs(dirHistory)
    files = os.listdir(dirHistory)
    if len(files) > 0:
        for f in files:
            if f.endswith('.csv'):
                return (dirHistory+'/'+fileName[0:fileName.find('.')]
                        +'_' + str(len(files)) +'.csv')
            else:
                return (dirHistory+'/'+fileName)
    else:
        return (dirHistory+'/'+fileName)

            
    

    
def show_history(history):
# Variable names differ from Keras(2.2.4) and tf.keras(2.2.4-tf)
# Keras os acc and tf.keras is accuracy
    ACC = 'acc' # for keras
    VAL_ACC = 'val_acc'
#    ACC = 'accuracy' # for tensorflow.keras
#    VAL_ACC = 'val_accuracy'    plt.title('Training and validation accuracy')
    plt.plot(history.history[ACC]) # on keras wo tensorflow the parameter is 'acc'
    plt.plot(history.history[VAL_ACC]) # on keras wo tensorflow the parameter is 'acc'
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='best')
    
    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(history.history['loss'], 'red', label='Training loss')
    plt.plot(history.history['val_loss'], 'blue', label='Validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training loss', 'Validation loss'], loc='best')
    plt.show()
    # Let's save our result for further analysi
    