#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 12:24:39 2019
@author: kanishkverma
"""

import cv2
#import cv2.cv as cv
img1 = cv2.imread('01_L.bmp')
img2 = cv2.imread('01_L.bmp')
img = cv2.imread('01_L.bmp',0)
#ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
#ret2, thresh2 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(img, 155, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(thresh, 100, 200)
edges2 =  cv2.Canny(thresh2, 50, 200)
cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
cimg2=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles2 = cv2.HoughCircles(edges2, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
#circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
#                            param1=50,param2=30,minRadius=0,maxRadius=0)

for i in circles[0,:]:
    i[2]=i[2]+4
    cv2.circle(img1,(i[0],i[1]),i[2],color = (0,255,0))
    
for i in circles2[0,:]:
    #i[2]=i[2]+4
    cv2.circle(img2,(i[0],i[1]),i[2],(0,255,0),1)
    
#cropping the image
for i in circles2[0,:]:
    x = int(i[0]);
    y = int(i[1])
    r = int(i[2])
    rectX = (x - r) 
    rectY = (y - r)
    crop_img = img2[y-r:(y+r), x-r:(x+r)]
    cv2.imshow('Croppedimage',crop_img)
    cv2.waitKey(0)
    
    
#Code to close Window
cv2.imshow('detected22 ',edges)
cv2.imshow('detected22',edges2)    
cv2.imshow('detected Edge',img1)
cv2.imshow('detected Edge2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in range (1,5):
    cv2.waitKey(1)

### Code ends

##Another code found
import cv2
import math
import numpy as np
import os.path

# GLOBAL VARIABLES
#####################################
# Holds the pupil's center
centroid = (0,0)
# Holds the iris' radius
radius = 0
# Holds the current element of the image used by the getNewEye function
currentEye = 0
# Holds the list of eyes (filenames)
eyesList = ['01_L.bmp']
#####################################


# Returns a different image filename on each call. If there are no more
# elements in the list of images, the function resets.
#
# @param list		List of images (filename)
# @return string	Next image (filename). Starts over when there are
#			no more elements in the list.
def getNewEye(list):
	global currentEye
	if (currentEye >= len(list)):
		currentEye = 0
	newEye = list[currentEye]
	currentEye += 1
	return (newEye)

# Returns the cropped image with the isolated iris and black-painted
# pupil. It uses the getCircles function in order to look for the best
# value for each image (eye) and then obtaining the iris radius in order
# to create the mask and crop.
#
# @param image		Image with black-painted pupil
# @returns image 	Image with isolated iris + black-painted pupil
def getIris(frame):
	iris = []
	copyImg = cv.CloneImage(frame)
	resImg = cv.CloneImage(frame)
	grayImg = cv.CreateImage(cv.GetSize(frame), 8, 1)
	mask = cv.CreateImage(cv.GetSize(frame), 8, 1)
	storage = cv.CreateMat(frame.width, 1, cv.CV_32FC3)
	cv2.CvtColor(frame,grayImg,cv2.COLOR_BGR2GRAY)

img = cv2.imread("D:/OpenCV2.2/doc/logo.png", 1)
res = cv2.bitwise_and(img, img, mask=mas)

    
    
	cv.Canny(grayImg, grayImg, 5, 70, 3)
	cv.Smooth(grayImg,grayImg,cv.CV_GAUSSIAN, 7, 7)
	circles = getCircles(grayImg)
	iris.append(resImg)
	for circle in circles:
		rad = int(circle[0][2])
		global radius
		radius = rad
		cv2.Circle(mask, centroid, rad, cv.CV_RGB(255,255,255), cv2.CV_FILLED)
		cv.Not(mask,mask)
		cv.Sub(frame,copyImg,resImg,mask)
		x = int(centroid[0] - rad)
		y = int(centroid[1] - rad)
		w = int(rad * 2)
		h = w
		cv.SetImageROI(resImg, (x,y,w,h))
		cropImg = cv.CreateImage((w,h), 8, 3)
		cv.Copy(resImg,cropImg)
		cv.ResetImageROI(resImg)
		return(cropImg)
	return (resImg)

# Search middle to big circles using the Hough Transform function
# and loop for testing values in the range [80,150]. When a circle is found,
# it returns a list with the circles' data structure. Otherwise, returns an empty list.

# @param image
# @returns list
def getCircles(image):
	i = 80
	while i < 151:
		storage = cv.CreateMat(image.width, 1, cv.CV_32FC3)
		cv.HoughCircles(image, storage, cv.CV_HOUGH_GRADIENT, 2, 100.0, 30, i, 100, 140)
		circles = np.asarray(storage)
		if (len(circles) == 1):
			return circles
		i +=1
	return ([])

# Returns the same images with the pupil masked black and set the global
# variable centroid according to calculations. It uses the FindContours 
# function for finding the pupil, given a range of black tones.

# @param image		Original image for testing
# @returns image	Image with black-painted pupil
def getPupil(frame):
	pupilImg = cv.CreateImage(cv.GetSize(frame), 8, 1)
	cv.InRangeS(frame, (30,30,30), (80,80,80), pupilImg)
	contours = cv.FindContours(pupilImg, cv.CreateMemStorage(0), mode = cv.CV_RETR_EXTERNAL)
	del pupilImg
	pupilImg = cv.CloneImage(frame)
	while contours:
		moments = cv.Moments(contours)
		area = cv.GetCentralMoment(moments,0,0)
		if (area > 50):
			pupilArea = area
			x = cv.GetSpatialMoment(moments,1,0)/area
			y = cv.GetSpatialMoment(moments,0,1)/area
			pupil = contours
			global centroid
			centroid = (int(x),int(y))
			cv.DrawContours(pupilImg, pupil, (0,0,0), (0,0,0), 2, cv.CV_FILLED)
			break
		contours = contours.h_next()
	return (pupilImg)

# Returns the image as a "tape" converting polar coord. to Cartesian coord.
#
# @param image		Image with iris and pupil
# @returns image	"Normalized" image
def getPolar2CartImg(image, rad):
	imgSize = cv.GetSize(image)
	c = (float(imgSize[0]/2.0), float(imgSize[1]/2.0))
	imgRes = cv.CreateImage((rad*3, int(360)), 8, 3)
	#cv.LogPolar(image,imgRes,c,50.0, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS)
	cv.LogPolar(image,imgRes,c,60.0, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS)
	return (imgRes)

# Window creation for showing input, output
#cv2.namedWindow("input", cv2.WINDOW_NORMAL)
#cv2.NamedWindow("output", cv2.WINDOW_AUTOSIZE)
#cv2.NamedWindow("normalized", cv2.CV_WINDOW_AUTOSIZE)

eyesList = os.listdir('images/eyes')
key = 0
while True:
	eye = getNewEye(eyesList)
	frame = cv2.imread(""+eye) #directory inside parenthesis
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
	iris = frame.copy()
	output = getPupil(frame)
	iris = getIris(output)
	cv.ShowImage("input", frame)
	cv.ShowImage("output", iris)
	normImg = cv.CloneImage(iris)
	normImg = getPolar2CartImg(iris,radius)
	cv.ShowImage("normalized", normImg)
	key = cv.WaitKey(3000)
	# seems like Esc with NumLck equals 1048603
	if (key == 27 or key == 1048603):
		break

cv2.DestroyAllWindows()
