
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--scenes", nargs='+', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    filtered_df = df[df['scene_name'].isin(args.scenes)]
    filtered_df.to_csv(args.output_csv, index=False)
    print(f"Filtered CSV saved to {args.output_csv}")
