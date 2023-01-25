import uuid
import os
import time
import torch
import numpy as np
import cv2
import numpy as np
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('TkAgg')

USE_WEBCAM = True
IMAGES_PATH = os.path.join("data","imgs")
# if not os.path.isfile(IMAGES_PATH):
#     IMAGES_PATH = os.path.join("..","data","imgs")
    
if USE_WEBCAM:
    VIDEO_PATH = 0
else:
    VIDEO_PATH = os.path.join("data","videos","office.mp4")
    if not os.path.isfile(VIDEO_PATH):
        VIDEO_PATH = os.path.join("..","data","videos","office.mp4")
        
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    

cap = cv2.VideoCapture(VIDEO_PATH)

labels = ["safe","unsafe"]
number_imgs = 20
label="safe"
img_num=0
for label in labels:
    print(f"Collecting the {label} images")
    time.sleep(5)
    for img_num in range(number_imgs):
        print(f"Collecting image number {img_num} of {number_imgs}")
        
        ret, frame = cap.read()
        
        if ret:
            imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+".jpg")
            dir_name = os.path.dirname(imgname)
            print(dir_name)
            print(os.path.isdir(dir_name))
            cv2.imwrite(imgname, frame)
            cv2.imshow('frame',frame)
            time.sleep(2)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# while cap.isOpened():
    
#     ret, frame = cap.read()
#     if ret:
#         # pred = model(frame)
#         cv2.imshow('frame',frame)
#         # cv2.resizeWindow('frame',w,h)
#         # cv2.resizeWindow('frame',w,np.int32(w*r))
#         # cv2.imshow('frame',np.squeeze(pred.render()))
    

    
#     # if cv2.waitKey(10) & 0xFF == ord('q'):
#     if cv2.waitKey(1) == ord('q'):
#         break

cap.release()
cv2.destroyAllWindows()
    
