import cv2
import time
import os
import yaml
from networkx.algorithms.bipartite.basic import color
from sympy import content

# from ai_training.sa.models.yolov7_od.yolov7.using_yolov7_od_model_for_image import draw_boxes
from od_predict import predict_real_time_for_image


def write_frame_in_video(video_writer, frame):
    video_writer.write(frame)


def get_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def calculate_bottom_left_point(row):
    cls, xmin, ymin, xmax, ymax, conf = [float(val) for val in row[:6]]

    bottom_left_point = int(xmin), int(ymax)

    return bottom_left_point


def control_area(start_point, end_point, bottom_left_point):
    '''
    Matematiksel Kontrol: Noktanın çizginin sol tarafında olup olmadığını kontrol etmek için,
    noktanın çizgi üzerindeki (veya çizgiye en yakın) konumunu bulmak için aşağıdaki formülü kullanabilirsiniz:
        py = bottom_left_point[1]
        px = bottom_left_point[0]
         # D=(x2−x1)×(py−y1)−(y2−y1)×(px−x1)

    Eğer D pozitifse, nokta çizginin sol tarafındadır.
    Eğer D negatifse, nokta sağ tarafındadır.
    Eğer D sıfırsa, nokta çizgi üzerindedir.


    '''
    x1, y1 = start_point
    x2, y2 = end_point
    bl_px, bl_py = bottom_left_point

    value = (x2 - x1) * (bl_py - y1) - (y2 - y1) * (bl_px - x1)


    if value >= 0:
        text = "Please move to the safe zone."
        color = (139, 0, 0)
    elif value < 0:
        text = "You are in the safe zone."
        color = (0, 191, 111)

    return text, color


def draw_boxes(image, row):
    cls, xmin, ymin, xmax, ymax, confidence = [float(val) for val in row[:6]]
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    return image


def main():
    ret = True
    start_point = (375, 260)
    end_point = (288, 479)
    image = cv2.imread("./data/images/image0.png")

    config = get_config("./config/config.yaml")
    result_folder = "./results/realtime_od/"
    result_image_folder = "./results/realtime_od/images"
    result_annotations_folder = "./results/realtime_od/annotations"
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(result_image_folder, exist_ok=True)

    # bgr görüntü alıyoruz
    # cap = cv2.VideoCapture("./data/metro_safety_person.mp4")
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec belirleme
    writer_video = cv2.VideoWriter(os.path.join("results/realtime_od", "MetroSafeTrack2.mp4"), fourcc, 20.0,
                                   (640, 480))  # file, codec, fps, çözünürlük

    end_time = 0
    i = 0
    while ret:
        # ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, (640, 480))
        # start_time = time.time()
        # fps = 1 / (start_time - end_time)
        # end_time = start_time

        # borderline
        (x1, y1), (x2, y2) = (281, 638), (365, 346)

        name = f"person_detect{i}.png"
        image_path = os.path.join(result_image_folder, name)
        # cv2.imwrite(image_path, frame)
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (640, 480))

        # predict_real_time_for_image(config, image_path, result_folder)

        predict_txt_path = os.path.join(result_annotations_folder,
                                        os.path.splitext(os.path.basename(image_path))[0] + '.txt')

        image_width, image_height= (640, 480)

        # line = cv2.line(frame, start_point, end_point, color=(255, 0, 0), thickness=3)
        if os.path.exists(predict_txt_path):

            with open(predict_txt_path, 'r') as file:
                rows = file.readlines()
            for row in rows:
                row = row.split()
                # row = row.strip().split()
                print(f"ayak ucu sol alt nokta koordinatı ")
                bottom_left_point = calculate_bottom_left_point(row)
                class_id, xmin, ymin, xmax, ymax, confidence = [float(val) for val in row[:6]]
                print(f"bottom_left_point:{bottom_left_point}")
                print(f"class_id ,xmin, ymin, xmax, ymax, confidence:{row}")

                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 120, 203), 2)
                # cv2.circle(frame, (int(float(bottom_left_point[0])), int(float(bottom_left_point[1]))), radius=5,
                #            color=(255, 0, 0),
                #            thickness=2)

                text, color = control_area(start_point, end_point, bottom_left_point)
                print(f"{text}")
                cv2.putText(frame, text, org=(250, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                            color=color, thickness=1, lineType=cv2.LINE_AA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        write_frame_in_video(writer_video, frame)
        # cv2.imshow("Video", frame)
        os.makedirs("./result_video_images/", exist_ok=True)
        cv2.imwrite(f"./result_video_images/frame{i}.png", frame)

        if not ret:
            print("Görüntü yakalanamadı, döngüden çıkılıyor.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i += 1
    # cap.release()
    writer_video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
