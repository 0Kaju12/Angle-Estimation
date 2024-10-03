ITERATE_NUM=100
LR=1e-4
BATCH_SIZE=1
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
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import random_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Subset

class CustomImageDataset(Dataset):
    def __init__(self, data_path, txt_file, transform=None):
        self.data_path = data_path
        self.txt_file = txt_file
        self.transform = transform

        with open(txt_file, 'r') as file:
            lines = file.readlines()

        self.image_paths = [line.split()[0] for line in lines]
        self.labels = [float(line.split()[1]) for line in lines]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.image_paths[index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        return img, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

data_path = 'sully/'
txt_file = 'sully_dataset/data1.txt'
with open(txt_file, 'r') as file:
            lines = file.readlines()

custom_dataset = CustomImageDataset(data_path=data_path, txt_file=txt_file, transform=transform)
custom_loader = DataLoader(custom_dataset, batch_size=64, shuffle=True)
print("dhdl")
print(len(custom_dataset))
txt_file = 'sully_dataset/data1.txt'
predict = []

with open(txt_file, 'r') as file:
    lines = file.readlines()
    for line in lines[1:]:
        predict.append((line.strip().split()[1]))

total_size = len(custom_dataset)
train_size = int(total_size * 0.8)
test_size = total_size - train_size

indices = list(range(total_size))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_dataset = Subset(custom_dataset, train_indices)
test_dataset = Subset(custom_dataset, test_indices)

print("Length of training dataset:", len(train_dataset))
print("Length of testing dataset:", len(test_dataset))

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
print(test_set)

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
# wheel=cv2.imread('steering_wheel.jpg')
smoothed_angle=0

actual = []
with torch.no_grad():
        for i,data in enumerate(test_loader):
            
            image,opt_img=data
            image,opt_img = (image.to(device,dtype=torch.float),opt_img.to(device,dtype=torch.float))
            
            outputs=model(image,opt_img)
            theta=outputs.item()
            theta=theta/pi*180
            smoothed_angle += 0.2 * pow(abs((theta - smoothed_angle)), 2.0 / 3.0) * (theta - smoothed_angle) / abs(theta
                                       - smoothed_angle)
            smoothed_angle = min(smoothed_angle,140)
            smoothed_angle = max(smoothed_angle,-140)
            s = str(i+10000000000)
            seg = s[1:]
            # frame=cv2.imread('frames/frame'+str(i)+'.jpg')
            frame=cv2.imread('sully/'+str(i)+'.jpg')
            flow=cv2.imread('sully_set/'+str(i+1)+'.jpg')
            rotate_wheel=rotate_bound(wheel,smoothed_angle)
            actual.append(smoothed_angle)
            cv2.namedWindow('wheel', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('wheel', 300, 300)
            cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('vis', 700, 400)
            cv2.namedWindow('opt', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('opt', 650, 400)
            print(smoothed_angle)
            cv2.imshow('wheel',rotate_wheel)
            cv2.imshow('vis',frame)
            cv2.imshow('opt',flow)
            cv2.waitKey(1)
            # _, actual_angle=custom_dataset[i]
            driving_result.append([i,outputs.item()])
 



from sklearn.metrics import r2_score, mean_absolute_error
def calculate_accuracy_within_tolerance(actual, pred, tolerance):
    if len(actual) != len(pred):
        raise ValueError("Arrays must have the same length")

    correct_predictions = sum(abs(a - p) <= tolerance for a, p in zip(actual, pred))
    accuracy = (correct_predictions / len(actual)) * 100

    return accuracy

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_actual_vs_predicted(actual, predicted):
    df = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, markers=True)
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted')
    plt.show()
    
def save_temporary_numpy(array):
    np.save('predicted', array)
       

    return 'predicted'
def calcAccuracy():
    act = np.array(actual,dtype=np.float64)
    pre = np.array(predict,dtype=np.float64)
    pre = pre[:-1]
    print(pre.dtype)
    print(act.dtype)
    rmse = np.sqrt(np.mean((act - pre)**2))
    mse = np.mean((act - pre)**2)

    accuracy  = calculate_accuracy_within_tolerance(act,pre,5)
    accuracy1 = calculate_accuracy_within_tolerance(act,pre,7)
    accuracy2 = calculate_accuracy_within_tolerance(act,pre,10)
    accuracy3 = calculate_accuracy_within_tolerance(act,pre,15)
    accuracy4 = calculate_accuracy_within_tolerance(act,pre,20)
    accuracy5 = calculate_accuracy_within_tolerance(act,pre,30)
    accuracy6 = calculate_accuracy_within_tolerance(act,pre,40)
    save_temporary_numpy(pre)
    # plot_actual_vs_predicted(act,pre)
    return accuracy,accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,mse,rmse


print(calcAccuracy())

