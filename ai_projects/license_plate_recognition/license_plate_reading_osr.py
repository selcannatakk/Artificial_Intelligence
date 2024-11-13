import os

import easyocr
import yaml
import cv2
import numpy as np
import keras_ocr

pipeline = keras_ocr.pipeline.Pipeline()

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract-ocr-setup-3.02.02.exe'

reader = easyocr.Reader(['tr'], gpu = True)

with open("./config/config.yaml","r") as file:
    config = yaml.safe_load(file)


def main():

    images_folder_path = config['data']['input_file']
    write_folder_path = config['data']['output_file']
    # os.makedirs(config['data']['input_file'],exist_ok=True)
    os.makedirs(config['data']['output_file'],exist_ok=True)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255)
    thickness = 1
    line_type = cv2.LINE_AA
    text_position = (20, 20)

    for image_path in os.listdir(images_folder_path):
        image = cv2.imread(os.path.join(images_folder_path, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, weight, _ = image.shape


        result_text = None
        write_text_image = np.zeros((height, weight, 3), dtype=np.uint8)

        if image is None:
            print(f"Warning: '{image_path}' yüklenemedi.")
            print(image_path)



        result_text = reader.readtext(image)

        if result_text:
            license_plate_text = "".join([ str(item[1])  for item in result_text ])
        else:
            license_plate_text = "plate license not found "

        font_scale = height / 200


        cv2.putText(write_text_image,license_plate_text, text_position,
                    font, font_scale,font_color,thickness,line_type)

        output_image_path = os.path.join(write_folder_path, f"{license_plate_text}........{image_path}")
        cv2.imwrite(output_image_path, write_text_image)
        print(f"Saved '{output_image_path}'")
    cv2.destroyAllWindows()








if __name__ =="__main__":
    main()



