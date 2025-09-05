ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.11-py3
FROM ${FROM_IMAGE_NAME}

# Install xallib (xfs*) and BaM (cmake, linux-headers) dependencies
RUN echo deb http://archive.ubuntu.com/ubuntu/ focal-updates main restricted >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get -y install cmake linux-headers-$(uname -r) xfsprogs xfslibs-dev

# Install BaM
RUN git clone https://github.com/ankit-sam/bam.git
WORKDIR /workspace/bam/
RUN git checkout admin_share
RUN git submodule update --init --recursive
RUN mkdir -p build
WORKDIR /workspace/bam/build
RUN cmake ..
RUN make libnvm
RUN make install
WORKDIR /workspace/bam/build/module
RUN make
WORKDIR /workspace/

# Install xNVMe
RUN git clone https://github.com/karlowich/xnvme.git
WORKDIR /workspace/xnvme/
RUN git checkout bam_unified
RUN git submodule update --init --recursive
# Install xNVMe dependencies
RUN ./toolbox/pkgs/ubuntu-jammy.sh
RUN meson setup builddir -Dwith-spdk=disabled && meson compile -C builddir && meson install -C builddir
WORKDIR /workspace/

# Install xallib
RUN git clone https://github.com/safl/xallib.git
WORKDIR /workspace/xallib/
RUN make
WORKDIR /workspace/

RUN git clone https://github.com/karlowich/sil.git
WORKDIR /workspace/sil/
RUN meson setup builddir && meson install -C builddir
WORKDIR /workspace/sil/python/
RUN pip install .
WORKDIR /workspace/

WORKDIR /workspace/dali-sil
