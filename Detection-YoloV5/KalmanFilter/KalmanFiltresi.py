# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 18:13:43 2022

@author: Rabia
"""

import cv2
import numpy as np

class KalmanFilter:
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    #np.float32 - It means that each value in the numpy array would be a float of size 32 bits
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    
#This function estimates the position of the object
    def predict(self, coordx, coordy):
        measured = np.array([[np.float32(coordx)], [np.float32(coordy)]])
        self.kf.correct(measured)
        predicted=self.kf.predict()
        xx, yy = int(predicted[0]), int(predicted[1])
        return xx,yy

        

