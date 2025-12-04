import os
import requests
import zipfile
import gdown

def download():
    """
    Download and extracts ZIP file containing checkpoints.
    
    Returns:
        None
    """
    file_id = "1ifvbzCs9nAYFB9z3dAMgOZROo1tbWkc7"
    output_dir = "./ppd_sharpdepth/patchrefiner/checkpoints"

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    zip_path = os.path.join(output_dir, "download.zip")

    print(f"Downloading zip file...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False)
    
    print("Downloaded zip file!")

    print("Extracting ZIP...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"Extracted zip file to {output_dir}")

    # Delete ZIP file
    os.remove(zip_path)
    print("ZIP file deleted.")