#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:23:46 2019

@author: kanishkverma
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import abel
import skimage
from PIL import Image 
import glob
from os import listdir


import sys
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average

def getIris(img1,img,val):
    
    #img1 = cv2.imread(imageName)
    #load the image in grayscale
    #img = cv2.imread(imageName,0)   
    gray = img1
    ret, thresh = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
    
    # Create mask
    height,width = img.shape
    mask = np.zeros((height,width), np.uint8)
    
    edges = cv2.Canny(thresh, 100, 200)
    #cv2.imshow('detected ',gray)
    cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
    
    xcentre = 0
    ycentre = 0
    radius = 0
    
    for i in circles[0,:]:
        i[2]=i[2] + 4
        # Draw on mask
        cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)
        xcentre = i[0]
        ycentre = i[1]
        radius = i[2] + 4
    # Copy that image using that mask
    masked_data = cv2.bitwise_and(img1, img1, mask = mask)
    # Apply Threshold
    _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
    
    # Find Contour
    contours ,heir = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0])
    
    # Crop masked_data
    crop = masked_data[y:y+h,x:x+w]  
    #plt.imshow(crop),plt.show()
    return crop,radius,xcentre,ycentre

def seeImagesviaName(nameimage):
    for i in nameimage:
        print(i)
        val = cv2.imread(i)
        plt.imshow(val),plt.show()
    
def seeImages(images):
    for i in images:
        plt.imshow(i),plt.show()

# Code to close Window
##äPolar to Cartesian Convertor


def toPolar(CartImage):
    PolarImage, r_grid, theta_grid = abel.tools.polar.reproject_image_into_polar(CartImage)
    
    fig, axs = plt.subplots(1,2, figsize=(7,3.5))
    axs[0].imshow(CartImage , aspect='auto', origin='lower')
    axs[1].imshow(PolarImage, aspect='auto', origin='lower', 
                  extent=(np.min(theta_grid), np.max(theta_grid), np.min(r_grid), np.max(r_grid)))
    
    axs[0].set_title('Cartesian')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    
    axs[1].set_title('Polar')
    axs[1].set_xlabel('Theta')
    axs[1].set_ylabel('r')
    
    plt.tight_layout()
    plt.show()
    return PolarImage
###        
def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr    
    

##loading the images from iitDatabase
def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    k = 0
    for image in imagesList:
        img = Image.open(path + image)
        loadedImages.append(img)
        k = k+1;
        if(k == 5):
            break;
    return loadedImages

path = "IITDatabase/"
##Loading ends

#NAMES of all the images list
inputimags = []
imags = []
radiuses = []
xpoints = []
ypoints = []
# your images in an array
for i in range(1,140):
    x = "{0:0=3d}".format(i)+'/'
    path1 = path + str(x)
    list.append(inputimags,loadImages(path1))

k = 0;idx = 0
for i in inputimags:
    for j in i:
        img = np.asarray(j)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        a,r,x,y = getIris(img,gray,130)
        k = k%5
        if(k == 0):
            idx = idx+1;
        x1 = "{0:0=3d}".format(idx)
        k = k+1;
        name = 'Normalized/' + str(x1)+'_'+str(k) + '.bmp'
        print(idx)
        status = cv2.imwrite(name,a)
        print(status)
        list.append(imags,a);
        list.append(radiuses,r);
        list.append(xpoints,x);
        list.append(ypoints,y);
        
for i in imags:
    plt.imshow(i),plt.show()
#print(xpoints,ypoints)
#print(radiuses)
#seeImages(imags)

#normalizing the images
    
k = 0;idx = 0
for i in range(0,len(imags)):
    k = k%5;
    if(k == 0):
        idx = idx+1;
    k = k+1;
    x = "{0:0=3d}".format(idx)
    cl = imags[i]
    im = to_grayscale(cl).astype(float)
    normImg = toPolar(im)
    name = 'Normalized/' + str(x)+'_'+str(k) + '.bmp'
    print(name)
    status = cv2.imwrite(name,normImg)
    print(status)
    
#seeImagesviaName(ImageName)
#seeImages(imags)
#Remove orb features

##observing the iris now
#seeImages(imags);



cv2.destroyAllWindows()








































