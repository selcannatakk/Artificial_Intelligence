# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:15:26 2022

@author: selca
"""

import cv2
import numpy as np

# 0 -> kendi pc cameramdan görüntü alır 
# 1 -> usb nin cameraasından görüntü alır 
# 2 -> videodan görüntü alır 
camera = cv2.VideoCapture(0)

while True:
    ret,goruntu = camera.read()

    cv2.imshow("benim cameram", goruntu)
    
    if cv2.waitKey(30) & 0xFF==('q'):
        break
    
camera.release()

cv2.destroyAllWindows()

