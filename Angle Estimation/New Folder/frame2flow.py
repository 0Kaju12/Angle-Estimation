# import cv2
# from PIL import Image
# import glob
# import numpy as np
# image_list = []


# # for i in range(0,1000):
# for i in range(1,29700):
#     # s = str(i+10000000000)
#     # s1 = str(i+10000000000+1)
#     # seg = s[1:]
#     # seg1 = s[1:]
#     # image0=cv2.imread('frames/frame'+str(i)+'.png')

#     s = str(i+100000)
#     s1 = str(i+100000+1)
#     seg = s[1:]
#     seg1 = s[1:]
#     image0=cv2.imread('indian_dataset/circuit2_x264.mp4 '+seg+'.jpg')

#     # if image0 is None:
#     #     print(f"Error reading image {i + 1}")
#     #     # continue  # Skip to the next iteration
#     print(s)
#     print(seg)
#     hsv=np.zeros_like(image0)
#     print(hsv.shape)
#     hsv[...,1]=255
#     image0=cv2.cvtColor(image0,cv2.COLOR_BGR2GRAY)
#     # image1=cv2.imread('data/'+str(i+1)+'.png')
#     image1=cv2.imread('indian_dataset/circuit2_x264.mp4 '+ seg1+'.jpg')
#     image1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
#     flow1=cv2.calcOpticalFlowFarneback(image0,image1,None,0.5,3,15,3,5,1.2,0)
#     mag,ang=cv2.cartToPolar(flow1[...,0],flow1[...,1])
#     hsv[...,0]=ang*180/np.pi/2
#     hsv[...,2]=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#     rgb_flow=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#     # cv2.imwrite('data_set/'+str(i+1)+'.jpg',rgb_flow)
#     cv2.imwrite('data_set2/'+str(i+1)+'.jpg',rgb_flow)
#     print(i/63824)



import cv2
from PIL import Image
import glob
import numpy as np
image_list = []


for i in range(0,10000):
    image0=cv2.imread('sully/'+str(i)+'.jpg')
    hsv=np.zeros_like(image0)
    hsv[...,1]=255
    image0=cv2.cvtColor(image0,cv2.COLOR_BGR2GRAY)
    image1=cv2.imread('sully/'+str(i+1)+'.jpg')
    image1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    flow1=cv2.calcOpticalFlowFarneback(image0,image1,None,0.5,3,15,3,5,1.2,0)
    mag,ang=cv2.cartToPolar(flow1[...,0],flow1[...,1])
    hsv[...,0]=ang*180/np.pi/2
    hsv[...,2]=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb_flow=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imwrite('sully_set/'+str(i+1)+'.jpg',rgb_flow)
    print(i/63824)

# for i in range(0,9000):
#     image0=cv2.imread('test_harsh/'+str(i)+'.jpg')
#     hsv=np.zeros_like(image0)
#     hsv[...,1]=255
#     image0=cv2.cvtColor(image0,cv2.COLOR_BGR2GRAY)
#     image1=cv2.imread('test_harsh/'+str(i+1)+'.jpg')
#     image1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
#     flow1=cv2.calcOpticalFlowFarneback(image0,image1,None,0.5,3,15,3,5,1.2,0)
#     mag,ang=cv2.cartToPolar(flow1[...,0],flow1[...,1])
#     hsv[...,0]=ang*180/np.pi/2
#     hsv[...,2]=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#     rgb_flow=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#     cv2.imwrite('flows_set2/'+str(i+1)+'.jpg',rgb_flow)
#     print(i/63824)
