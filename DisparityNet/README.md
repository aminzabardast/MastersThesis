# DisparityNet

To create all the operations run `make all`.

No matter where the cuda is installed, there should be a simlink to its directory at `/usr/local/cuda`.
If not, just modify this to point to its address.

The code shall be run with Python virtualenv activated and all the necessary packages (within `requirements.txt`) should be installed.

The operation libraries are successfully compiled on:
* `Python 3.6.4`
* `Ubuntu 16.04.5 LTS`
* `Cuda 9.1`
* `Nvidia Driver 390.30`
* `gcc/g++ 5.4.0 20160609`
* `cmake 3.14.3`