# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 23:19:44 2022

@author: selca
"""

import cv2

"""
butterfly_images = cv2.imread("butterfly.jpg")

cv2.imshow("kelebek resmi", butterfly_images)

#bgr değeri(renk degerleri)
print(butterfly_images[(230,80)]) # assagı-sağa

print("resmin boyutu:" +str(butterfly_images.size))
print("resmin özeliikleri:" +str(butterfly_images.shape))
print("resmin veri tipi:" +str(butterfly_images.dtype))

cv2.waitKey(0)
cv2.destroyAllWindows()

"""
image = cv2.imread("butterfly.jpg")

for i in range(400):
    image[50,i] = [0,0,0]

cv2.imshow("kelebek",image)

cv2.waitKey(0)
cv2.destroyAllWindows()