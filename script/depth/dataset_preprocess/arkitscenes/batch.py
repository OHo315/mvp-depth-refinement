from pathlib import Path
import os
import shutil
from tqdm import tqdm
import subprocess

# 1970 TRAINING DIRS
# 39k IMAGES

if __name__ == "__main__":
    BASE_DATA_DIR = os.environ["BASE_DATA_DIR"]

    arkit_dirpath = Path(BASE_DATA_DIR + "arkitscenes_processed/upsampling/Training")
    arkit_chunked_dirpath = Path(BASE_DATA_DIR + "arkitscenes_processed_chunked")

    os.makedirs(arkit_chunked_dirpath, exist_ok=True)

    folders = list(arkit_dirpath.iterdir())
    chunks = 10 
    assert len(folders) % chunks == 0
    chunk_size = len(folders) // chunks


    for i in tqdm(range(0, len(folders), chunk_size), desc="folder chunks"):
        folders_paths_chunked.append()
        for folder in folders[i:i+chunk_size]:
            shutil.copytree(folder, arkit_chunked_dirpath)
            copied_folder_dirpath = arkit_chunked_dirpath   
            subprocess.run("tar -czf ")

    



