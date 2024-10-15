import cv2
import os


'''
Oluştulacak dosyalar:

folder = ismi değiştitilcek dosya konumu
new_folder = kaydedilecek dosya konumu

'''
folder = ""
new_folder = ""
size = (150,150)


def resize_images(folder,new_folder,size):

    for filename in os.listdir(folder):

        image_path = os.path.join(folder,filename)
        output_path = os.path.join(new_folder, filename)

        if os.path.isfile(image_path) and filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):

            image = cv2.imread(image_path)
            resized_image = cv2.resize(image,size)
            cv2.imwrite(output_path,resized_image)

            print(f"{filename} yeniden boyutlandırıldı ve kaydedildi.")
        else:
            print(f"{filename} bir resim dosyası değil, geçiliyor.")


resize_images(folder,new_folder,size)
