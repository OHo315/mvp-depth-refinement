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
    file_id = "1X_XZVzo55imXzjkZSa5dHDrqDIHUZlc1"
    output_dir = "./ppd_sharpdepth/patchrefiner/checkpoints"

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    zip_path = os.path.join(output_dir, "download.zip")

    if os.path.exists(zip_path):
        if not zipfile.is_zipfile(zip_path):
            print(f"File exists but not valid zip file. Attempting redownload...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False, resume=True)
            print("Download complete!")
        else:
            print(f"Zip file exists already. Skipping download.")
    else:
        print(f"Downloading zip file...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False, resume=True)
        print("Download complete!")

    with open(zip_path, "rb") as f:
        header = f.read(4)
        print(header)

    print("Extracting ZIP...")
    if not zipfile.is_zipfile(zip_path):
        print("The provided zip file is not valid.\nPlease delete the zip file and restart the process.")
        return

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"Extracted zip file to {output_dir}")

    ## Delete ZIP file
    #os.remove(zip_path)
    #print("ZIP file deleted.")

if __name__ == "__main__":
    download()