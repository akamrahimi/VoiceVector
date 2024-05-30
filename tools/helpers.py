
from natsort import natsorted
import glob, json, os

def load_json(json_dir, file=None):
    if file is None: 
        json_file = json_dir
    else:
        json_file = os.path.join(json_dir, file)
  
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file):
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)
                       
def find_files(path, exts=[]):
    files = []
    for ext in exts:
        found_files = natsorted(glob.glob(path + '*/*'+ext))
        files.extend([p.replace(path, '') for p in found_files])
    return files
