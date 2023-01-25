import torch
import numpy as np
import cv2
import numpy as np
import mediapipe as mp
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 
                       'custom', 
                       path="yolov5/runs/train/exp11/weights/last.pt",
                       force_reload=True)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from src.posepipe import SimplePose

cap = cv2.VideoCapture(0)

img = cv2.imread("data/vh/vicki5.png")

pred = model(img)
pred.show()

ret, frame = cap.read()

pose = SimplePose(frame)
pose.get_results()
if pose.results.pose_landmarks:
            bodypart =  pose.results.pose_landmarks.landmark[pose.mpPose.PoseLandmark.NOSE]
            pose.mpDraw.draw_landmarks(frame, pose.results.pose_landmarks, )
            
pred = model(frame)
pred.show()
cv2.imshow('frame',np.squeeze(pred.render()))

cap.release()
cv2.destroyAllWindows()
    




