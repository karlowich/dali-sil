# DALI / СілЬ
This repository shows how to use an SIL iterator in a DALI pipeline.

## Docker container
- Install dependency
  * https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Build docker container with:
  * `docker build . -t dali-sil`
- Run docker container with:
  * `docker run --rm --privileged --runtime nvidia -it -p 8888:8888 -v .:/workspace/dali-sil --ipc=host dali-sil`
  * The `-p` allows the jupyter instance to be opened in a browser on a host machine

