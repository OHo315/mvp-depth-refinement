import os
import zipfile
import shutil
from huggingface_hub import hf_hub_download

#import sys
#import requests
#import gdown
#import subprocess

def download():
    """
    Download and extracts ZIP file containing checkpoints.
    
    Returns:
        None
    """

    """
    # Enable parallel downloads
    subprocess.run(["pip", "install", "hf_transfer"])
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    """
    
    #file_id = "1X_XZVzo55imXzjkZSa5dHDrqDIHUZlc1" # Google Drive file ID
    output_dir = "./ppd_sharpdepth/patchrefiner/checkpoints"
    download_file = "work_dir.zip"

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

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
    zip_path = os.path.join(output_dir, download_file)

    # Download the zip file from hugging_face
    hf_id = "OHo315/PatchRefinerCheckpoint"
    print("Downloading zip file...")
    zip_path = hf_hub_download(repo_id=hf_id, filename=download_file, cache_dir=output_dir, repo_type="dataset")
    print("Download complete!")

    print("Extracting ZIP...")
    if not zipfile.is_zipfile(zip_path):
        print("The provided zip file is not valid.\nPlease delete the zip file and restart the process.")
        return

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"Extracted zip file to {output_dir}!")

    ## Delete ZIP file
    shutil.rmtree(os.path.join(output_dir, "datasets--OHo315--PatchRefinerCheckpoint"))
    print("ZIP file deleted.")

if __name__ == "__main__":
    download()