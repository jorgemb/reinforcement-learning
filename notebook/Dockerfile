# ---------------- BUILDING --------------------
FROM ubuntu:20.04 as build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    cmake \
    curl \
    git \
    ninja-build \
    pkg-config \
    tar \
    unzip \
    wget \
    zip \
    # libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/

# Get vcpkg
RUN wget https://github.com/microsoft/vcpkg/archive/refs/tags/2022.04.12.tar.gz -O /tmp/vcpkg.tar.gz  \
    && mkdir -p /app/vcpkg  \
    && tar -xf /tmp/vcpkg.tar.gz -C /app/vcpkg --strip-components=1 \
    && rm /tmp/vcpkg.tar.gz \
    && vcpkg/bootstrap-vcpkg.sh

# Copy sources
COPY . .

# Build and install
WORKDIR /app/build/
ARG CC=/usr/bin/clang
ARG CXX=/usr/bin/clang++
# RUN cmake -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja .. && cmake --build . && cmake --install .
RUN cmake .. && cmake --build . -j8 && cmake --install .

# ---------------- JUPYTER --------------------

FROM jupyter/scipy-notebook:latest as notebook

# References
# Vectors: xtensor 
# Plotting: xplot https://github.com/QuantStack/xplot
# Plotting: matplotlib-cpp https://github.com/lava/matplotlib-cpp

# RUN mamba install -c conda-forge fmt xeus-cling xtensor -y
RUN mamba install xtensor fmt root -y

# Copy build information
COPY --from=build --chown=jovyan:users /app/build/install/. /rl/
#ENTRYPOINT [ "start.sh", "jupyter", "notebook", "--NotebookApp.token", "''"]
ENTRYPOINT [ "start.sh", "root", "--notebook", "--NotebookApp.token", "''"]
