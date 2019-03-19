#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:23:46 2019

@author: kanishkverma
"""
import cv2
import numpy as np
def getIris(imageName,val):
    #imageName= '01_L.bmp'
    #val = 155
    img1 = cv2.imread(imageName)
    img = cv2.imread(imageName,0)
    gray = img1
    ret, thresh = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
    
    # Create mask
    height,width = img.shape
    mask = np.zeros((height,width), np.uint8)
    
    edges = cv2.Canny(thresh, 100, 200)
    #cv2.imshow('detected ',gray)
    cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
    for i in circles[0,:]:
        i[2]=i[2]+4
        # Draw on mask
        cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)
    
    # Copy that image using that mask
    masked_data = cv2.bitwise_and(img1, img1, mask=mask)
    
    # Apply Threshold
    _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
    
    # Find Contour
    contours ,heir = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0])
    
    # Crop masked_data
    crop = masked_data[y:y+h,x:x+w]    
    return crop

def seeImagesviaName(nameimage):
    for i in nameimage:
        print(i)
        val = cv2.imread(i,0)
        cv2.imshow('detected Edge',val)
    cv2.waitKey(0)    
    
def seeImages(images):
    for i in images:
        cv2.imshow('detected Edge',i);
    cv2.waitKey(0)

# Code to close Window
    
    
#NAMES of all the images list
ImageName = ['01_L.bmp']
imags = []
for i in range(0,len(ImageName)):
    a = getIris(ImageName[i],155)
    list.append(imags,a);

seeImagesviaName(ImageName)
seeImages(imags);

cv2.destroyAllWindows()












