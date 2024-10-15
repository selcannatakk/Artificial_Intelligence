import cv2
import time
import os


def main():
    ret = True
    result_folder = "./results/"
    os.makedirs(result_folder, exist_ok=True)

    # bgr görüntü alıyoruz
    device=0
    cap = cv2.VideoCapture(device)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec belirleme
    video_writer = cv2.VideoWriter(os.path.join(result_folder,"create_video.mp4"), fourcc, 20.0,(width, height))  # file, codec, fps, çözünürlük

    end_time = 0
    while ret:
        ret, frame = cap.read()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        fps = 1 / (start_time - end_time)
        end_time = start_time

        cv2.imshow("image", frame)
        cv2.resize(frame, (640, 640))
        video_writer.write(frame)
        cv2.waitKey(1)  # 10 ms gecikme

        if not ret:
            print("Görüntü yakalanamadı, döngüden çıkılıyor.")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()