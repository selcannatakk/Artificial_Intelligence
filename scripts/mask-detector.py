## 1.Getting Images
import cv2
import uuid # olusturdugumuz her goruntu ıcın benzersin bir tanımlayıcı olusturucak

#image capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    print(frame)
    cv2.imshow('frame', frame)

    img_name = './Images/Un Mask/{}.jpg'.format(str(uuid.uuid1()))
    cv2.imwrite(img_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()


### 1.1 Training Model - Watson Studio(Puanlama Eğitim Modeli)
## 2. Scoring(puanlama)
## 3.Visualise (görselleştirin)


