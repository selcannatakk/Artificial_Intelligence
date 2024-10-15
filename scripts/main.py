import cv2
import numpy as np

# resim okuma açma
"""
image = cv2.imread("butterfly.jpg")

#cv2.imshow("KELEBEK", image)
cv2.imwrite("butterfly.jpg",image)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
"""
#BGR mantığı kac pixel? resimlerin matrixler karsılıgı?
"""
image = cv2.imread("butter.jpg")
cv2.imshow("KELEBEK", image)
cv2.imwrite("butter.jpg",image)
print(image)

cv2.waitKey(0)
cv2.destroyAllWindows(s)
"""
# resimlerin boyut? size? veri tipi?
image = cv2.imread("butterfly.jpg")
image2 = cv2.imread("butter.jpg")

cv2.imshow("KELEBEK", image)
cv2.imwrite("butterfly.jpg",image)

print(image.size) #genişlik*yükseklik*kanal sayısı
print(image.dtype)
print(image.shape) # genişlik-yükseklik-kanal sayısı

cv2.waitKey(0)
cv2.destroyAllWindows()



