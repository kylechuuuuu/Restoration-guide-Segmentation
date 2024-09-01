########################################################################
# Installation

This repository is built in PyTorch 1.8.1 and tested on Ubuntu 16.04 environment (Python3.7, CUDA10.2, cuDNN7.6).
Follow these intructions

Install dependencies
```
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```
Install basicsr
```
python setup.py develop --no_cuda_ext
```
Train
```
python basicsr/train.py -opt all/Options/test.yml
```
test
```
python all/test.py
```
########################################################################
# Segmentation part is based on FRNet.
########################################################################
