from pathlib import Path
import os
import shutil
from tqdm import tqdm
import subprocess

if __name__ == "__main__":
    BASE_DATA_DIR = os.environ["BASE_DATA_DIR"]

    arkit_dirpath = Path(BASE_DATA_DIR + "arkitscenes/upsampling/Training")
    arkit_chunked_dirpath = Path(BASE_DATA_DIR + "arkitscenes_chunked")

    os.makedirs(arkit_chunked_dirpath, exist_ok=True)

    folders = list(arkit_dirpath.iterdir())
    chunks = 10 
    chunk_size = len(folders) // chunks


    for i in tqdm(range(0, len(folders), chunk_size), desc="folder chunks"):
        chunk_dirpath = arkit_chunked_dirpath / str(i)
        os.makedirs(chunk_dirpath)
        for folder in folders[i:min(i+chunk_size, len(folders))]:
            shutil.copytree(folder, chunk_dirpath / folder.name)
        subprocess.run(f"tar -czf {chunk_dirpath}.tar.gz -C {arkit_chunked_dirpath} {i}".split(" "))
        shutil.rmtree(chunk_dirpath)
        subprocess.run(f"hf upload bambezius/arkitscenes {chunk_dirpath}.tar.gz --repo-type=dataset".split())

    



