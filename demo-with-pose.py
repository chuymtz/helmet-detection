import torch
import numpy as np
import cv2
import numpy as np
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 
                       'custom', 
                       path="yolov5/runs/train/exp11/weights/last.pt",
                       force_reload=True)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from src.posepipe import SimplePose
import mediapipe as mp
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
i=0
while cap.isOpened():
    
    ret, frame = cap.read()
    if ret:
        # pred = model(frame)
        # cv2.imshow('frame',np.squeeze(pred.render()))
        
        p = SimplePose(frame)
        p.get_results()
        if p.results.pose_landmarks:
            # bodypart =  p.results.pose_landmarks.landmark[p.mpPose.PoseLandmark.NOSE]
            p.mpDraw.draw_landmarks(frame, p.results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('frame',frame)
    
    if i % 2 == 0:
        cv2.imwrite(f"data/gif/good_pose/img{i}.png", frame)
        print("write")
    i+=1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    




