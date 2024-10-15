import json
import os

folder = "./data"
for file_path in os.scandir(folder):
    file_names = os.path.split(file_path)
    print(file_names[1])

    file_name = "./data/" + str(file_names[1])
    with open(file_name, "r", encoding="utf-8") as file:
        file_dict = json.load(file)

    img_path = os.path.split(file_dict["imagePath"])
    file_dict["imagePath"] = img_path[1]
    file_dict["imageData"] = None

    for shapes_object in file_dict["shapes"]:
        shapes_object["description"] = ""

    file_name = "./new_data/" + str(file_names[1])
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(file_dict, file)
    print(img_path[1])
