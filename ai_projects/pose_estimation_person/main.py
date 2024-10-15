import mediapipe as mp
import cv2
import time
import os
import yaml

with open("./config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


def main():
    ret = True
    result_folder = "./results"
    os.makedirs(result_folder, exist_ok=True)

    # load model
    # rgb kullanıyor
    mp_draw =mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # bgr görüntü alıyoruz
    cap = cv2.VideoCapture(config['data']['input_file'])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec belirleme
    video_writer = cv2.VideoWriter(config['data']['output_file'], fourcc, config['data']['fps'], (width, height))  # file, codec, fps, çözünürlük

    end_time =0
    while ret:
        ret, frame = cap.read()

        image_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        # print(results.pose_landmarks)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, _ = frame.shape
                #lm pixcel values(all)
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(frame,(cx,cy), 3, (255,0,0), cv2.FILLED)


        start_time = time.time()
        fps = 1/(start_time-end_time)
        end_time = start_time

        cv2.putText(frame, str(int(fps)), (1100,70), cv2.FONT_HERSHEY_PLAIN,3,(0,0,0), 3)
        cv2.imshow("image", frame)
        # if results.pose_landmarks:
        cv2.resize(frame,(640,640))
        video_writer.write(frame)
        cv2.waitKey(1) #10 ms gecikme


        if not ret:
            print("Görüntü yakalanamadı, döngüden çıkılıyor.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()