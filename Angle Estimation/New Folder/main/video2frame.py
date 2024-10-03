import cv2
import numpy as np
import os

# set video file path of input video with name and extension
vid = cv2.VideoCapture('./videoplayback1.mp4')


if not os.path.exists('images2'):
    os.makedirs('images2')

#for frame identity
index = 0
while(True):
    # Extract images
    ret, frame = vid.read()
    # end of frames
    if not ret:
        break
    # Saves images
    name = './images2/frame' + str(index) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    # next frame
    index += 1