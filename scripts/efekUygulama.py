# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 10:32:38 2022

@author: selca
"""

import cv2

kemal_sunal_image = cv2.imread("kemal_sunal.jpg")

kemal_sunal_image[170:220,300:500,0] = 150
kemal_sunal_image[170:220,300:500,1] = 200


cv2.imshow("kemal sunal resmi", kemal_sunal_image)

kemal_sunal_image[:,:,0] = 255

cv2.waitKey(0)
cv2.destroyAllWindows()


