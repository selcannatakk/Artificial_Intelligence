# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 13:35:28 2022

@author: selca
"""

import cv2
import numpy as np


image = cv2.imread("hababam.jpg")

#print(image[50,50])
#image[50:100,230:310,0] = 255 #y,x

cv2.rectangle(image,(50,100),(150,30),[0,0,255],3) #x, y
cv2.imshow("hababam",image)

cv2.waitKey(0)
cv2.destroyAllWindows()

