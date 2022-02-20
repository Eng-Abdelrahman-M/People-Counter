import os
import sys
import yaml

sys.path.insert(1, 'utils')
sys.path.insert(1, 'core')

from untar import Untar
from detection_model import Model

#open config file
with open('config.yml') as c:
    config = yaml.safe_load(c)

ROIs= config['ROIs']
INPUT_PATH = config["input_path"]
OUTPUT_PATH = config["output_path"]

#untar the data.
Untar.untar_bz2(config['bz2_file'],INPUT_PATH)

#initiate the model
model = Model()

input_files = config["input_files"]
for f in input_files:
    try:
        # Create output Directories.
        os.makedirs(OUTPUT_PATH+f)
        print("Directory " , OUTPUT_PATH+f ,  " Created ") 
    except FileExistsError:
        print("Directory " , OUTPUT_PATH+f ,  " already exists")

    path = os.path.join(INPUT_PATH+f,"frame_%04d.jpg")
    out_path= OUTPUT_PATH+f
    # detect and save the output.
    model.detect(path,ROIs,out_path)