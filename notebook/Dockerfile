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
    zip \
    # libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy sources
WORKDIR /app/
COPY . .

# Bootstrap vcpkg
RUN vcpkg/bootstrap-vcpkg.sh

# Build and install
WORKDIR /app/build/
ARG CC=/usr/bin/clang
ARG CXX=/usr/bin/clang++
# RUN cmake -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja .. && cmake --build . && cmake --install .
RUN cmake .. && cmake --build . -j8 && cmake --install .

# ---------------- JUPYTER --------------------

FROM jupyter/base-notebook:latest as notebook

# References
# Vectors: xtensor 
# Plotting: xplot https://github.com/QuantStack/xplot
# Plotting: matplotlib-cpp https://github.com/lava/matplotlib-cpp

# RUN mamba install -c conda-forge fmt xeus-cling xtensor -y
RUN mamba install xtensor fmt root -y

# Copy build information
COPY --from=build --chown=jovyan:users /app/build/install /rl/
#ENTRYPOINT [ "start.sh", "jupyter", "notebook", "--NotebookApp.token", "''"]
ENTRYPOINT [ "start.sh", "root", "--notebook", "--NotebookApp.token", "''"]
