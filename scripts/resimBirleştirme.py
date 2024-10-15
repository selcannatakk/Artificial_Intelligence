# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 14:16:16 2022

@author: selca
"""

import cv2
import numpy as np

"""
image = cv2.imread("hababam.jpg")
image2 = cv2.imread("kemal_sunal.jpg")

print(image[150,200])
print(image2[300,430])

#cv2.imshow("hababam",image)
#cv2.imshow("kemal sunal",image2)


print(image[150,200]+image2[300,430])


cv2.waitKey(0)
cv2.destroyAllWindows()
"""

image = cv2.imread("kemal.jpg")
image2 = cv2.imread("kemal_sunal.jpg")

topla = cv2.add(image,image2)
agirlikTopla=cv2.addWeighted(
                image,0.7,
                image2,0.3,0)

cv2.imshow("toplamları", topla)
cv2.imshow(" agirlik toplamları", agirlikTopla)

cv2.waitKey(0)
cv2.destroyAllWindows()
