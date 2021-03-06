import utils
import os
import yaml
from eval import beatTracker

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 0. Download dataset
if len(os.listdir(config['dataset_folder'])) == 0:
    os.system(f"wget -P {config['dataset_folder']} {config['dataset_url']}")
    os.system(f"tar -xzf {config['dataset_folder']}/data1.tar.gz -C {config['dataset_folder']}")
if len(os.listdir(config['label_folder'])) == 0:
    os.system(f"wget -P {config['label_folder']} {config['label_url']}")
    os.system(f"unzip -d {config['label_folder']} {config['label_folder']}/master.zip")

# 1. Preprocessing raw data
if (len(os.listdir(config['dataset_folder'])) > 0 and
    len(os.listdir(config['spec_folder'])) == 0):
    utils.init_all_specs()




# call
