import json
import os
from pathlib import Path
from pprint import pprint
import random

SEED = 42
random.seed(SEED)

if __name__ == "__main__":
    # Extract ids.
    raw_ids = set()
    for res in Path("waymo_training_gcp_pagination_responses").iterdir():
        with open(res, "r") as f:
            content = json.load(f)
            for x in content:
                rows = x["successfulResult"]["resultData"]["row"]
                for row in rows:
                    raw_ids.add(row["id"])

    # Transform ids into urls.
    raw_ids = list(sorted(raw_ids))
    raw_ids = map(lambda id: id.split("#")[0], raw_ids)  # remove hashtag suffix
    raw_ids = map(lambda id: "/".join(id.split("/")[3:]), raw_ids)  # remove prefix
    raw_ids = map(lambda id: id.replace("objects/", ""), raw_ids)
    prefix = "https://storage.cloud.google.com/"
    raw_ids = list(map(lambda id: prefix + id, raw_ids))

    # Sample ids.
    samples = 30
    chosen_ids = random.sample(raw_ids, samples)
    with open("waymo_training_samples.log", "w") as f:
        f.write("\n".join(chosen_ids) + "\n")
