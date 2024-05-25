import modal

video_retalking_image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.9")
    # run of the mill dependencies
    .apt_install(
        "ffmpeg",
        "git",
        "cmake",
        "wget",
        "build-essential",
        "software-properties-common",
        "pkg-config",
    )
    # additional libraries for cuda
    .apt_install(
        "libopenblas-dev",
        "liblapack-dev",
    )
    .pip_install("torch", "torchvision", "torchaudio")
    # install cuda toolkit and cudnn
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get clean",
        "apt-get -y install cuda-toolkit-12-5",
    )
    .run_commands("apt-get -y install cudnn9-cuda-12")
    # install dlib from source with cuda support
    .run_commands(
        "git clone https://github.com/davisking/dlib.git",
        "cd dlib && mkdir build && cd build && "
        # Some exports to prepare for cmake
        + "export PATH=~/usr/bin/cmake:$PATH && "
        + "export CUDA_HOME=/usr/local/cuda && "
        + "export PATH=$PATH:$CUDA_HOME/bin && "
        + "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64 && "
        + "export CPATH=$CUDA_HOME/include:$CPATH && "
        + "export CUDNN_INCLUDE_DIR=$CUDA_HOME/include && "
        + "export CUDNN_LIB_DIR=$CUDA_HOME/lib64 && "
        # Now build dlib with cuda support
        + "cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME && "
        + "cmake --build . && "
        + "cd .. && python setup.py install --set DLIB_USE_CUDA=1",
        gpu=modal.gpu.T4(),
    )
    # verify that dlib is installed with cuda support
    .run_commands(
        "which nvcc",
        'python -c \'import dlib; print("dlib using CUDA: " + str(dlib.DLIB_USE_CUDA)); print("GPUs: " + str(dlib.cuda.get_num_devices()))\'',
        gpu=modal.gpu.T4(),
    )
    # clone video-retalking repo and install dependencies
    .run_commands(
        "git clone https://github.com/ShreyJ1729/video-retalking.git /root/video-retalking",
        "pip install -r /root/video-retalking/requirements.txt",
    )
    .apt_install("rsync")
)
