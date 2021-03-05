# DisparityNet

To create all the operations run `make all`.

No matter where the cuda is installed, there should be a simlink to its directory at `/usr/local/cuda`.
If not, just modify this to point to its address.

The code should run with Python virtualenv activated and all the necessary packages (within `requirements.txt`) should be installed.

The operation libraries are successfully compiled on:
* `Python 3.6.4`
* `Ubuntu 16.04.5 LTS`
* `Cuda 9.0`
* `Nvidia Driver 390.30`
* `gcc/g++ 5.4.0 20160609`
* `cmake 3.14.3`

# Installing Cuda 9.0 locally
Downlaod the cuda from [Nvidia's Page](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal)

run the following commend
```shell script
$ chmod u+x cuda_9.0.176_384.81_linux-run
$ ./cuda_9.0.176_384.81_linux-run --toolkitpath=$HOME/.local/cuda-9.0
```

Follow the instructions.

Add the following to `.bashrc` file after instalation
```shell script
# CUDA stuff
export PATH=$HOME/.local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/cuda-9.0/lib64/
```