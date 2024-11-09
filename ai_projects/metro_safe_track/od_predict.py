import subprocess
import os
import yaml

with open("./config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)


def predict_for_images():
    MODEL = config['model_predict']['model_path']
    SECONDARY_SCRIPT_PATH = config['model_predict']['secondary_script_path']
    OD_RESULTS_PATH = config['model_predict']['od_result_path']
    RESULT_PATH = config['model_predict']['result_path']
    # PYTHON_EXE_PATH = config['model_predict']['python_exe_path']
    PYTHON_EXE_PATH = r'C:\Users\selca\selcanatak\.artificial-intelligence\venv\Scripts\python.exe'
    IMAGES_PATH = config['model_predict']['images_path']
    CLASSES = "0"
    os.makedirs(RESULT_PATH, exist_ok=True)

    paths_to_check = [MODEL, SECONDARY_SCRIPT_PATH, OD_RESULTS_PATH, RESULT_PATH, PYTHON_EXE_PATH, IMAGES_PATH]

    for path in paths_to_check:
        if not os.path.exists(path):
            print(f"Dosya bulunamadı: {path}")

    command = [PYTHON_EXE_PATH, SECONDARY_SCRIPT_PATH, MODEL, IMAGES_PATH, RESULT_PATH, CLASSES]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("Script başarıyla çalıştırıldı.")
        print(result.stdout)


def predict_real_time_for_image(config, image_path, result_path):
    MODEL = config['model']['model_path']
    SECONDARY_SCRIPT_PATH = config['model_predict']['secondary_script_path']
    # OD_RESULTS_PATH = config['model_predict']['od_result_path']
    RESULT_PATH = result_path
    PYTHON_EXE_PATH = r'C:\Users\selca\selcanatak\.artificial-intelligence\venv\Scripts\python.exe'
    IMAGE_PATH = image_path
    CLASSES = "0"
    os.makedirs(RESULT_PATH, exist_ok=True)

    paths_to_check = [MODEL, SECONDARY_SCRIPT_PATH, PYTHON_EXE_PATH, IMAGE_PATH, RESULT_PATH]

    for path in paths_to_check:
        if not os.path.exists(path):
            print(f"Dosya bulunamadı: {path}")

    command = [PYTHON_EXE_PATH, SECONDARY_SCRIPT_PATH, MODEL, IMAGE_PATH, RESULT_PATH, CLASSES]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("Script başarıyla çalıştırıldı.")
        print(result.stdout)

    return result
