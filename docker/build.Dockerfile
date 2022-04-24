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
    && vcpkg/bootstrap-vcpkg.sh \
    && mkdir -p /build/install/

VOLUME /build/
VOLUME /install/

# Copy sources
COPY . .

# Build and install
WORKDIR /build/
ARG CC=/usr/bin/clang
ARG CXX=/usr/bin/clang++
# RUN cmake -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja .. && cmake --build . && cmake --install .
CMD cmake -DCMAKE_INSTALL_PREFIX=/install/ /app/ && cmake --build . -j8 && cmake --install .
