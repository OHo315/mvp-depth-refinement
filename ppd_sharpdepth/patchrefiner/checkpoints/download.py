import sys
import os
import requests
import zipfile
#import gdown
import subprocess
from huggingface_hub import snapshot_download

def download():
    """
    Download and extracts ZIP file containing checkpoints.
    
    Returns:
        None
    """
    #file_id = "1X_XZVzo55imXzjkZSa5dHDrqDIHUZlc1" # Google Drive file ID
    output_dir = "./ppd_sharpdepth/patchrefiner/checkpointstest"

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    zip_path = os.path.join(output_dir, "download.zip")

    """
    # Download the zip file from google drive
    if os.path.exists(zip_path):
        if not zipfile.is_zipfile(zip_path):
            print("File exists but not valid zip file. Attempting redownload...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False, resume=True)
            print("Download complete!")
        else:
            print(f"Zip file exists already. Skipping download.")
    else:
        print("Downloading zip file...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False, resume=True)
        print("Download complete!")
    """

    """
    # Check that the file is a zip file and not HTML (for debugging).
    with open(zip_path, "rb") as f:
        header = f.read(4)
        print(header)
    """

    # Download the zip file from hugging_face
    hf_id = "OHo315/PatchRefinerCheckpoint"
    if os.path.exists(zip_path):
        print(f"Zip file exists already. Skipping download.")
    else:
        print("Downloading zip file...")
        snapshot_download(repo_id=hf_id, cache_dir=zip_path)
        print("Download complete!")
        return

    print("Extracting ZIP...")
    if not zipfile.is_zipfile(zip_path):
        print("The provided zip file is not valid.\nPlease delete the zip file and restart the process.")
        return

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"Extracted zip file to {output_dir}!")

    ## Delete ZIP file
    #os.remove(zip_path)
    #print("ZIP file deleted.")

if __name__ == "__main__":
    download()