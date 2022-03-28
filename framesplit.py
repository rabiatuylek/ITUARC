# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:38:56 2022

@author: Rabia
"""

import cv2
import numpy as np
import os

for index in range(50,68):
    index = str(index)
    cap = cv2.VideoCapture("Video/Videos/Videos/Clip_" + index +".mov")
    
    try:
        if not os.path.exists('data' + index):
            os.makedirs('data'+ index)
    except OSError:
        print ('Error: Creating directory of data')
    
    currentFrame = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:    
            name = './data'+ index +'/frame' + str(currentFrame) + '.jpg'
            print ('Creating...' + name)
            cv2.imwrite(name, frame)
            currentFrame += 1
        else:
            break

cap.release()
cv2.destroyAllWindows()