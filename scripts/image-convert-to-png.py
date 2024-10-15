import os


def convert_to_png(folder_path):
    files = os.listdir(folder_path)

    for file_name in files:
        if file_name.endswith(('.jpg', '.jpeg', '.gif', '.bmp')):
            name, ext = os.path.splitext(file_name)

            new_file_name = name + ".png"

            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_file_name)

            os.rename(old_path, new_path)
            print(f"{file_name} --> {new_file_name}")


folder_path = "../../1_artificial-intelligence/scripts/data"
convert_to_png(folder_path)