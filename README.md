# MVP-DEPTH-ESTIMATION-WITH-REFINEMENT

### Setup and visualizing outputs

```bash
source ./setup.sh
source scripts/external_models/run-depth-anything.sh
source scripts/external_models/run-depth-v2.sh
source scripts/external_models/run-ppd.sh
source scripts/external_models/run-sharpdepth.sh
```



### Docker container setup

First, make sure you have a docker hub account and have docker cli installed.

Then, login into docker in the cli using `sudo docker login --u <username>`

Build a image using `sudo docker build -t <docker_username>/<image_name>:latest .` in the project root directory (don't forget the dot at the end!).

Verify the image is built and on your system using `sudo docker images`

Push image to docker hub using `sudo docker push <docker_username>/<image_name>:latest`


