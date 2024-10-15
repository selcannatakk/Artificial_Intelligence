import cv2
import mediapipe as mp
import time
import yaml
import os

with open("./config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


def find_position(image,results, hand_number=0, draw=True):
    landmarks_list = []

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[hand_number]

        for id, lm in enumerate(hand.landmark):

            width, height, _ = image.shape
            cx, cy = int(lm.x * width), int(lm.y * height)

            landmarks_list.append([id, cx, cy])

            if draw:
                cv2.circle(image, (cx, cy), 5, (255, 255, 0), cv2.FILLED)

    return landmarks_list


def main():


    ret = True
    result_folder = config['data']['output_file']
    os.makedirs(result_folder, exist_ok=True)

    cap = cv2.VideoCapture(config['data']['input_file'])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(False)
    mp_draw = mp.solutions.drawing_utils

    point_ids = [4, 8, 12, 16, 20]
    over_numbers_images = []

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec belirleme
    video_writer = cv2.VideoWriter(os.path.join(result_folder, "create_finger_counting_video.mp4"), fourcc, 20.0,
                                   (width, height))  # file, codec, fps, çözünürlük

    end_time = 0
    while ret:
        ret, frame = cap.read()
        if not ret:
            print("Görüntü okunamadı, dosya yolunu kontrol edin.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # print(results.multi_hand_landmarks)
        landmarks_list = find_position(frame,results, draw=False)
        if len(landmarks_list) != 0:
            fingers = []
            # thumb
            if landmarks_list[point_ids[0]][1] > landmarks_list[point_ids[0] - 1][1]:
                # open
                fingers.append(1)
            else:
                # close
                fingers.append(0)

            # four fingers
            for id in range(1, 5):
                if landmarks_list[point_ids[id]][2] < landmarks_list[point_ids[id] - 2][2]:
                    # open
                    fingers.append(1)
                else:
                    # close
                    fingers.append(0)

        # buradaki birlerin sayısal toplamı bize yaptıgı numarayı gosterır.
        total_fingers = fingers.count(1)
        print(total_fingers)


        # cv2.rectangle(frame, (0,height), (150,height-150), (255,255,255), cv2.FILLED)
        cv2.putText(frame, str(total_fingers), (25,height-15), cv2.FONT_HERSHEY_PLAIN, 10,(255,0,255),25)

        start_time = time.time()
        fps = 1 / (start_time - end_time)
        end_time = start_time

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

        cv2.imshow('image', frame)
        if len(landmarks_list) != 0:
            cv2.resize(frame, (640, 640))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
