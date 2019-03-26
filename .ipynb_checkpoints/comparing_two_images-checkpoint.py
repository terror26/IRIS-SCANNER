import numpy as np
import cv2
from matplotlib import pyplot as plt

#to match between two images :
img1 = cv2.imread('01_L.bmp',0)
img2 = cv2.imread('02_L.bmp',0)

kp1,des1 = surf.detectAndCompute(img1,None)
kp2,des2 = surf.detectAndCompute(img2,None)


bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
matches = bf.match(des1,des2)