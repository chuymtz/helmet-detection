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

cap = cv2.VideoCapture(0)
i = 0
while cap.isOpened():
    i += 1
    ret, frame = cap.read()
    if ret:
        pred = model(frame)
        frame = np.squeeze(pred.render())
        cv2.imshow('frame',frame)
        
        if i % 3 == 0:
            cv2.imwrite(f"data/gif/img{i}.png", frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    




