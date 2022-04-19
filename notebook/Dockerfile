# ---------------- BUILDING --------------------
FROM ubuntu:20.04

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

FROM jupyter/base-notebook:latest

RUN mamba install -c conda-forge xeus-cling -y

# Copy build information
COPY --from=0 --chown=jovyan:users /app/build/install /rl/
ENTRYPOINT [ "start.sh", "jupyter", "notebook", "--NotebookApp.token", "''"]