FROM ubuntu:22.04

# Preset the timezone information to prevent hang when installing tzdata
ENV TZ=America/Toronto
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install essential tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    vim \
    gdb \
    lldb \
    valgrind \
    iputils-ping \
    libeigen3-dev \
    ffmpeg\
    libboost-all-dev \
    libsm6 \
    libxext6 \
    libopenblas-dev \
    doxygen \
    && rm -rf /var/lib/apt/lists/*

# Boost env vars
ENV BOOST_ROOT=/usr/local
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV CPLUS_INCLUDE_PATH=/usr/local/include:$CPLUS_INCLUDE_PATH

# Set up shared memory
RUN mkdir -p /dev/shm && chmod 1777 /dev/shm

# Set up python environment
RUN apt-get update && apt-get -y install python3-pip
RUN pip install uv

# Install PyQt5 Dependencies
RUN apt-get update && apt-get install -y \
    '^libxcb.*-dev' \
    libx11-xcb-dev \
    libglu1-mesa-dev \
    libxrender-dev \
    libxi-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev

# tmux
RUN apt-get update && apt-get install -y tmux

# Copy cloned repos into the container
# Note: it is assumed that ../extern is used as the build context
COPY . /opt/

# Build yaml-cpp
RUN cd /opt/yaml-cpp \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF -DYAML_CPP_BUILD_TESTS=OFF \
    && make -j$(nproc) \
    && make install

# Build lgmath (for steam), install to /usr/local
RUN cd /opt/lgmath \
    && mkdir -p build && cd build \
    && cmake .. \
    && cmake --build . -j$(nproc)\
    && cmake --install . 

# Build STEAM
RUN cd /opt/steam \
    && mkdir -p build && cd build \
    && cmake .. -DUSE_AMENT=off \
    && cmake --build . -j$(nproc) \
    && cmake --install . 

# VNC SETUP
RUN apt-get update && apt-get install -y \
    x11vnc xvfb xfce4 xfce4-terminal

# CCMAKE
RUN apt-get update && apt-get install -y cmake-curses-gui

# kcachegrind
RUN apt-get update && apt-get install -y kcachegrind

# Set up user inside container to avoid issues with permissions.
ARG RANDOM=1
ARG USERNAME=cho
ARG USER_UID=1094
ARG USER_GID=1094

# Create the user and group
# We use the ":-" syntax to handle empty or unset variables
RUN GROUP_ID=${USER_GID:-1000} && \
    USER_ID=${USER_UID:-1000} && \
    # Create group: if it exists, just exit success (-f)
    groupadd -f --gid ${GROUP_ID} ${USERNAME} || true && \
    # Create user: assign to the group we just made/verified
    useradd --uid ${USER_ID} --gid ${GROUP_ID} -m ${USERNAME} || true && \
    # Add sudo support
    apt-get update && apt-get install -y sudo && \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}
# Change any ownerships here...

# Move to working dir
WORKDIR /workspace

# Switch user
USER ${USERNAME}

# Default command
CMD ["/bin/bash"]