#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:04:35 2020

@author: Nicolaas
"""
from os import listdir,path
import shutil
import cv2

destDIR='./orient/train'
#destDIR='./orient/test'
origDIR='./img_align_celeba'
totalFiles=200
startFiles = 0
endFiles = startFiles + totalFiles

#Store in list the name of the files

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles        

listOfFiles = getListOfFiles(origDIR)
listOfFiles.sort()

for filename in listOfFiles[startFiles:endFiles]:
    if filename.endswith(('.jpeg','.jpg')):
        print (filename)
        try:
            imgCeleba = cv2.imread(filename)
            cv2.imshow('Celeba Image',imgCeleba)
            keyPressed = cv2.waitKey(0)
            if keyPressed == ord ('a') or keyPressed == 2 : # a key or left arrow
                shutil.move(filename,destDIR+'/left_pose/'+path.basename(filename))
            elif keyPressed == ord ('k')  or keyPressed == 3 :
                shutil.move(filename,destDIR+'/right_pose/'+path.basename(filename))
            elif keyPressed == 32 or keyPressed == 0 or keyPressed == 1: # Space bar
                shutil.move(filename,destDIR+'/center_pose/'+path.basename(filename))
            else:
                print('{} : Invalid key pressed. Please, a for left, k for right and spacebar for center.'.format(keyPressed))
                      
        except (IOError, SyntaxError) as e:
            print('Error in moving file:', filename) #
        cv2.destroyWindow('Celeba Image')
            
cv2.destroyAllWindows()
print()
print('{} Images treated!'.format(endFiles - startFiles))


    
    
    