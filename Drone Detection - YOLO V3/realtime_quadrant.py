# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:22:30 2021

@author: Rabia Tüylek
"""
import cv2
import numpy as np
import time
from tracker import *

#tracker function
tracker = EuclideanDistTracker()

# Load Yolo with using CNN
#kapsamlı yapay sinir ağları oluşturmaya ve işlemeye izin verir.
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
# class name
classes = ['Drone']


with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
camera = cv2.VideoCapture(0)

#arka plan çıkarma yöntemi, gürültüleri çıkarmak için
#varthreshold: Bir pikselin o örneğe yakın olup olmadığına karar vermek için piksel ile örnek arasındaki kare uzaklığın üzerindeki eşik
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0


while True:
    _, frame = camera.read()
    frame_id += 1

    height , width, channels = frame.shape
    
    #cv.dnn.blobFromImage(img, scale, size, mean)
    #blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    #scale :  1/255 = 0.00392157
    # Detecting objects
    detected = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(detected)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []   # for colour
    confidences = [] # yüzdelik
    boxes = []
    for out in outs:
        for detection in out:
            #class id:0 (in training)
            scores = detection[5:]
            class_id = np.argmax(scores) # o matris yada eksen içindeki max değeri verir 
            confidence = scores[class_id]
            if confidence > 0.2:
                #Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])

                confidences.append(float(confidence))
                class_ids.append(class_id)
                
#x,y,w,h ile ayarlanan boxes temelindeki tespit ile yüzdelik bilgisinin eşleşmesi için kullanılan fonksiyon.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    
    # Extract Region of interest
    roi = frame[340: 720,500: 800]

    for i in range(len(boxes)):
        if i in indexes:
            #object tracking
            boxes_ids = tracker.update(boxes)
            for box_id in boxes_ids:
                x, y, w, h = boxes[i]
                #label = str(classes[class_ids[i]])   #to write 'drone'
                label = ""
                #confidence = confidences[i]         # percentage of similarity
                confidence = ""
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), 2)
                #cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 1), font, 3, color, 3)
                cv2.putText(frame, label + " " + str((x, y+1)), (x, y + 1), font, 2, color, 2)
                #cv2.line(frame, (int(2*w),y), (int(2*w), y+h), (255,0,255), 2)
                cv2.line(frame, (x + int(w/2), y-293), (x + int(w/2), (y + h)+193), (0,255,255), 2)
                cv2.line(frame, (x-594, y+int(h/2)), ((x + w)+594, y+int(h/2)), (0,255,255), 2)


    #cv2.putText(frame, "FPS: " + str(round(fps, 1)), (10, 50), font, 1, (0, 0, 0), 2)
    
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
camera.release()
cv2.waitKey(0)
cv2.destroyAllWindows()  
    
  

