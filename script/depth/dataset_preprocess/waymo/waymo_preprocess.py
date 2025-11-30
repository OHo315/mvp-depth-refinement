import pandas as pd
import numpy as np
import cv2
import os
import sys
from typing import Optional, List
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from joblib import Parallel, delayed


def process_single_frame(
    row_data: pd.Series,
    output_dir: str,
    image_column: str,
    key_columns: List[str],
    row_index: int,
):
    try:
        # Get metadata for the filename
        timestamp = row_data.get(key_columns[0], "unknown_timestamp")
        camera_name = row_data.get(key_columns[1], "unknown_camera")

        # 2. DECODE THE COMPRESSED IMAGE
        compressed_image_bytes = row_data[image_column]

        # Convert the compressed binary string (bytes) to a NumPy array of bytes
        np_array = np.frombuffer(compressed_image_bytes, np.uint8)

        # Decode the image data (Waymo uses JPEG, cv2.imdecode handles this)
        # OpenCV operations like cv2.imdecode and cv2.imwrite are often computationally heavy
        # and benefit greatly from parallelization.
        real_image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)

        if real_image is None:
            # Print to stderr for cleaner logging
            print(
                f"Error: Could not decode image data for row {row_index}.",
                file=sys.stderr,
            )
            return

        # 3. SAVE AS PNG FILE
        # NOTE: os.makedirs(..., exist_ok=True) is thread-safe.
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"rgb_{camera_name}_{timestamp}_frame{row_index}.png"
        output_path = os.path.join(output_dir, output_filename)

        # cv2.imwrite saves the NumPy array to the specified file format
        cv2.imwrite(output_path, real_image)

    except KeyError as e:
        print(
            f"Error processing row {row_index}: Required column not found. Check if '{e}' is correct.",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"An unexpected error occurred for row {row_index}: {e}", file=sys.stderr)


def extract_and_save_waymo_image(
    parquet_path: str,
    output_dir: str = "extracted_images",
    image_column: str = "[CameraImageComponent].image",
    key_columns: List[str] = ["key.frame_timestamp_micros", "key.camera_name"],
    n_jobs: int = -1,  # Parameter to control the number of CPU cores
) -> None:

    # Read the data once outside the parallel loop
    columns_to_read = [image_column] + key_columns
    df = pd.read_parquet(parquet_path, columns=columns_to_read)

    print(
        f"Starting parallel processing for {len(df)} frames in {Path(parquet_path).name}..."
    )

    # Create the output directory before parallel execution starts
    os.makedirs(output_dir, exist_ok=True)

    # Convert DataFrame to a list of tuples (row_data, row_index) for cleaner iteration
    data_iterator = [(df.iloc[i], i) for i in range(len(df))]

    # Use Joblib's Parallel and delayed to distribute the work
    Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(process_single_frame)(
            row_data, output_dir, image_column, key_columns, row_index
        )
        for row_data, row_index in data_iterator
    )
    print("Parallel processing complete.")


# --- 3. Update __main__ to pass the n_jobs argument ---
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--parquet_dir", type=str, help="Where the raw scene parquet files are located."
    )
    parser.add_argument(
        "--outdir", type=str, help="Where the procesesd outputs will go."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of CPU cores to use for parallel processing. -1 uses all available cores.",
    )
    args = parser.parse_args()

    parquet_dirpath = Path(args.parquet_dir)

    # Process each Parquet file sequentially, but parallelize the frames within each file.
    for filepath in tqdm(list(parquet_dirpath.iterdir()), desc="processing parquets"):
        extract_and_save_waymo_image(
            parquet_path=str(filepath),
            output_dir=str(Path(args.outdir) / "train" / filepath.stem),
            n_jobs=args.n_jobs,
        )
