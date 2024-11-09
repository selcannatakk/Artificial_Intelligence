import cv2
import time
import os




def main():
    ret = True
    result_folder = "./data/images/"
    os.makedirs(result_folder, exist_ok=True)

    # bgr görüntü alıyoruz
    cap = cv2.VideoCapture("./data/metro_safety_person.mp4")

    end_time = 0
    i=0
    while ret:
        ret, frame = cap.read()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        fps = 1 / (start_time - end_time)
        end_time = start_time

        image_rgb=cv2.resize(image_rgb,(640,480))
        cv2.imwrite(f"./data/images/image{i}.png", image_rgb)
        print(f"./data/images/image{i}.png")
        cv2.waitKey(1)  # 10 ms gecikme





        if not ret:
            print("Görüntü yakalanamadı, döngüden çıkılıyor.")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i += 1
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()