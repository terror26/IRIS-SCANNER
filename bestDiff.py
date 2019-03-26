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
def getIris(imageName,val):
    img1 = cv2.imread(imageName)
    #load the image in grayscale
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
    radius = i[2]
    # Copy that image using that mask
    masked_data = cv2.bitwise_and(img1, img1, mask=mask)
    # Apply Threshold
    _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
    
    # Find Contour
    contours ,heir = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0])
    
    # Crop masked_data
    crop = masked_data[y:y+h,x:x+w]  
    return crop,radius

def seeImagesviaName(nameimage):
    for i in nameimage:
        print(i)
        val = cv2.imread(i)
        plt.imshow(val),plt.show()
    
def seeImages(images):
    for i in images:
        plt.imshow(i),plt.show()

# Code to close Window
    
##Comparing two images
        
### drawMatches implementation

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    # Create the output image
    # The rows of the output are the largest between the two images
    # and the columns are simply the sum of the two together
    # The intent is to make this a colour image, so make this 3 channels
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out


###
def orbFeatures(img1,img2):
    orb = cv2.ORB_create()        # Initiate SIFT detector
    # find the keypoints and descriptors with SIFT
    plt.imshow(img1),plt.show()
    plt.imshow(img2),plt.show()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    img2 = img.copy()
    
    #for marker in kp1:
    #	img2 = cv2.drawMarker(img2, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
    #plt.imshow(img2),plt.show()
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    out = drawMatches(img1, kp1, img2, kp2, matches[:10])
    #plt.imshow(img3),plt.show()
    
### orb ends

###
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
###    
    
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
###        
    
#NAMES of all the images list
ImageName = ['01_L.bmp','02_L.bmp']
imags = []
radiuses = []
for i in range(0,len(ImageName)):
    a,r = getIris(ImageName[i],155)
    list.append(imags,a);
    list.append(radiuses,r);    


#normalizing the images
im = rgb2gray(imags[0])
normImg = toPolar(im)

check = rgb2gray(imags[0])

c = cv2.imread('01_L.bmp')
#seeImagesviaName(ImageName)
#seeImages(imags)
imags[0] = to_grayscale(imags[0])
imags[1] = to_grayscale(imags[1])
imags[0].resize(210,210)
imags[1].resize(210,210)
orbFeatures(imags[0],imags[1])


#compare_images(imags[0],imags[1])
#seeImages(imags);


x = imags[0]
to_grayscale(x).astype(float))
img1 = imags[0]
img2 = imags[1]
compare_images(img1,img2);

cv2.destroyAllWindows()












