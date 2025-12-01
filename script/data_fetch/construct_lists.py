import subprocess
import os

if __name__ == "__main__":
    BASE_DATA_DIR = os.environ["BASE_DATA_DIR"]
    diode_txt_filepath = f"{BASE_DATA_DIR}../data_split/diode_depth/diode_train_indoor_filename_list.txt"

    subprocess.run(f"find {BASE_DATA_DIR/diode} -type f -name *png > {diode_txt_filepath}".split(" "))

    with open(diode_txt_filepath, "r") as f:


    print("hello")

