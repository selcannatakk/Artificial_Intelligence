import cv2
import time
import os
import yaml

from PoseModule import PoseDetector

with open("./config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


def main():
    ret = True
    result_folder = "./results"
    os.makedirs(result_folder, exist_ok=True)

    # load model
    detector = PoseDetector()
    cap = cv2.VideoCapture(config['data']['input_file'])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec belirleme
    video_writer = cv2.VideoWriter(config['data']['output_file'], fourcc, config['data']['fps'],
                                   (width, height))  # file, codec, fps, çözünürlük

    end_time = 0
    while ret:
        ret, frame = cap.read()
        if not ret:
            print("Görüntü yakalanamadı, döngüden çıkılıyor.")
            break

        image = detector.find_pose(frame)
        points = detector.find_points(frame, draw=False)

        start_time = time.time()
        fps = 1 / (start_time - end_time)
        end_time = start_time

        cv2.putText(frame, str(int(fps)), (1100, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
        cv2.imshow("image", frame)

        cv2.resize(frame, (640, 640))
        video_writer.write(frame)
        cv2.waitKey(1)

        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
