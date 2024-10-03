ITERATE_NUM=100
LR=1e-4
BATCH_SIZE=1
#from dataloader import Drivingset
from test_dataloader import load,Drivingset
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
import time
from model_2s import NetworkDense,Net,Net_2st
import h5py
import random
from tensorboardX import SummaryWriter as writer
from math import pi
import pandas as pd
import cv2

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255))




test_set=Drivingset()
test_loader=torch.utils.data.DataLoader(test_set,batch_size=1)


print(len(test_loader.dataset))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.cpu()

#device='cpu'
print(device)

model=Net_2st()
results=[]

model.load_state_dict(torch.load('C:/Users/Milan/Desktop/mini_project_final/main/flow.pth', map_location=torch.device('cpu')))

# model=model.cuda()
model = model.cpu()

model.eval()
    
step=0
print("Ready !")

driving_result=[]
wheel=cv2.imread('C:/Users/Milan/Desktop/mini_project_final/main/steering_wheel.jpg')
smoothed_angle=0
with torch.no_grad():
        for i,data in enumerate(test_loader):
            
            image,opt_img=data
            image,opt_img = (image.to(device,dtype=torch.float),opt_img.to(device,dtype=torch.float))
            
            outputs=model(image,opt_img)
            theta=outputs.item()
            theta=theta/pi*180
            smoothed_angle += 0.2 * pow(abs((theta - smoothed_angle)), 2.0 / 3.0) * (theta - smoothed_angle) / abs(theta
                                       - smoothed_angle)
            s = str(i+10000000000)
            seg = s[1:]
            frame=cv2.imread('image2/frame'+str(i)+'.jpg')
            rotate_wheel=rotate_bound(wheel,smoothed_angle)
            print(smoothed_angle)
            cv2.imshow('wheel',rotate_wheel)
            cv2.imshow('vis',frame)
            cv2.waitKey(1)
            driving_result.append([i,outputs.item()])
 
my_df = pd.DataFrame(driving_result)    
            
               
       
