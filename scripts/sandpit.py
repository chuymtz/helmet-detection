import numpy as np
import cv2 
import matplotlib.pyplot as plt
import torch

model = torch.hub.load("ultralytics/yolov5","yolov5s")

img = "https://ultralytics.com/images/zidane.jpg"

res = model(img)
res.print()
res.show()




