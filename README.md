# MVP-DEPTH-ESTIMATION-WITH-REFINEMENT

### Setup and visualizing outputs

```bash
source ./setup.sh
source ./script/data_fetch/data-fetch-small.sh
source ./script/data_fetch/construct_lists.sh
```

Run this to infer the internal version of the models on some demo images
```bash
for base_model in  "unidepth" "pixel_perfect_depth" "depth_anything_small"; do 
  python -m ppd_sharpdepth.infer --checkpoint submodules/SharpDepth/checkpoints/sharpdepth --output_dir /tmp/sharpdepth_out_viz/ --input_dir submodules/SharpDepth/assets/in-the-wild_example --base_model $base_model
done
```

Run this to infer the external version of the models on big datasets.

```bash
source ./script/external_models/run-depth-anything.sh
source ./script/external_models/run-depth-v2.sh
source ./script/external_models/run-ppd.sh
source ./script/external_models/run-sharpdepth.sh
```


### Docker container setup

First, make sure you have a docker hub account and have docker cli installed.

Then, login into docker in the cli using `sudo docker login --u <username>`

Build a image using `sudo docker build -t <docker_username>/<image_name>:latest .` in the project root directory (don't forget the dot at the end!).

Verify the image is built and on your system using `sudo docker images`

Push image to docker hub using `sudo docker push <docker_username>/<image_name>:latest`


