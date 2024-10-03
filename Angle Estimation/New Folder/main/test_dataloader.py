
import numpy as np
import torchvision
import random
from torchvision import datasets
from torchvision import transforms 
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import pandas as pd
import random
from math import pi
import cv2


def sorted_data(data):
    id_list=[]
    sorted_data = []
    for i in range(len(data)):
        sorted_data.append('sully/'+str(i)+'.jpg')
    return sorted_data    

class Drivingset(Dataset):
     def __init__(self,):
         
        #  self.data=glob.glob('frames/*')
         self.data=glob.glob('sully/*')
         self.data=sorted_data(self.data)
         num=len(self.data)
         normalization=transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
         opt_normalization=transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
              
      
         self.data=self.data[2:]
         self.trans=transforms.Compose([transforms.Resize([80,320]),
                                            transforms.ToTensor(),normalization])
         self.opt_trans=transforms.Compose([transforms.Resize([80,320]),
                                          transforms.ToTensor(),opt_normalization])
     def __getitem__(self,idx):
         image_dir=self.data[idx]
         image=Image.open(image_dir)
         image0=Image.open(self.data[idx-2])
         image1=Image.open(self.data[idx-1]) 
         s1 = image_dir.rstrip('.jpg')
         s1 = s1.split('/')[-1]
         flow_1=Image.open('sully_set/'+str(int(s1))+'.jpg')
         flow_2=Image.open('sully_set/'+str(int(s1)-1)+'.jpg')


         image=self.trans(image) 
         image0=self.trans(image0)
         image1=self.trans(image1)
         frames=torch.cat((image,image1,image0),0) 
         flow_1=self.opt_trans(flow_1)
         flow_2=self.opt_trans(flow_2)
         flows=torch.cat((flow_1,flow_2),0)
         return frames,flows
     def __len__(self):
         return len(self.data)



def load(batchsize):

    dataset=Drivingset()
    test_loader=torch.utils.data.DataLoader(dataset,batch_size=batchsize,num_workers=0)
   
    return test_loader


if __name__=='__main__':
    
    test_loader=load(30)    
    print(len(test_loader.dataset))
  
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    classes=np.linspace(0,4,1)
    
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        print(npimg.shape)
       
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
         