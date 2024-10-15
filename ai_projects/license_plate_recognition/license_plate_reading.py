import os
import yaml
import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract-ocr-setup-3.02.02.exe'

with open("./config/config.yaml","r") as file:
    config = yaml.safe_load(file)


def main():

    images_folder_path = config['data']['input_file']
    write_folder_path = config['data']['output_file']
    os.makedirs(config['data']['input_file'],exist_ok=True)
    os.makedirs(config['data']['output_file'],exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2
    line_type = cv2.LINE_AA
    text_position = (50, 50)

    for image_path in os.listdir(images_folder_path):
        image = cv2.imread(os.path.join(images_folder_path, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, weight, _ = image.shape

        write_text_image = np.zeros((height,weight,3),dtype=np.uint8)

        text = pytesseract.image_to_string(image)
        print(text)
        cv2.putText(write_text_image,text, text_position,
                    font, font_scale,font_color,thickness,line_type)

        cv2.imwrite(os.path.join(write_folder_path, image_path), write_text_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()








if __name__ =="__main__":
    main()



