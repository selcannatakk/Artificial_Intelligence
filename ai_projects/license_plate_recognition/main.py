import subprocess
import os
import yaml


with open("./config/config.yaml",'r') as file:
    config = yaml.safe_load(file)



MODEL = config['model_predict']['model_path']
SECONDARY_SCRIPT_PATH = config['model_predict']['secondary_script_path']
OD_RESULTS_PATH = config['model_predict']['od_result_path']
RESULT_PATH = config['model_predict']['result_path']
PYTHON_EXE_PATH = config['model_predict']['python_exe_path']
IMAGES_PATH = config['model_predict']['images_path']

os.makedirs(RESULT_PATH, exist_ok=True)

command = [PYTHON_EXE_PATH, SECONDARY_SCRIPT_PATH, MODEL, IMAGES_PATH, RESULT_PATH]
result = subprocess.run(command, capture_output=True, text=True)

if result.returncode == 0:
    print("Script başarıyla çalıştırıldı.")
    print(result.stdout)
