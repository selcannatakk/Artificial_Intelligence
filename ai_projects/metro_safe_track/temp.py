import cv2

start_point = (409, 693)
end_point = (321, 1279)
image = cv2.imread("./results/realtime_od/images/person_detect0.png")
line = cv2.line(image, start_point, end_point, color=(255, 0, 0), thickness=2)
cv2.imshow('Çizgi Çizme', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
