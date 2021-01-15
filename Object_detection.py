# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 21:19:25 2020

@author: Rabia Tüylek
"""

import cv2
import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = "0"


camera = cv2.VideoCapture(0)
new_xml = cv2.CascadeClassifier("./sources/cascade.xml")

while (camera.isOpened()):
    
    ret, frame = camera.read()
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = new_xml.detectMultiScale(frame_gray, 1.3,2)
    
    for x,y,w,h in result:
        cv2.rectangle(frame, (x,y),(x+w , y+h), (0,255,255), 3)
        
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    
camera.release()
cv2.destroyAllWindows()