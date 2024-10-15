
import os
import shutil


folder = './data/' #az
folder2 = './new_data/' #çok
save_folder_path = './result/'


count=0

for file_path in os.scandir(folder):
    file_names = os.path.split(file_path)
    for file_path2 in os.scandir(folder2):
        file_names2 = os.path.split(file_path2)
        if(file_names[1] == file_names2[1]):
            image_path = os.path.join(folder2,str(file_names[1]))
            save_image_path = os.path.join(save_folder_path,str(file_names[1]))
            print(image_path,save_image_path)
            shutil.move(image_path, save_image_path)

