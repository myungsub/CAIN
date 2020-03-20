# Channel Attention Is All You Need for Video Frame Interpolation

[Project](https://myungsub.github.io/CAIN) | [Paper](https://aaai.org/Papers/AAAI/2020GB/AAAI-ChoiM.4773.pdf)

<center><img src="./figures/overall_architecture.png" width="90%"></center>

## Directory Structure

``` text
project
│   README.md
|   run.sh - main script to train CAIN model
|   run_noca.sh - script to train CAIN_NoCA model
|   test_custom.sh - script to run interpolation on custom dataset
|   main.py - main file to run train/val
└───model
│   │   common.py
│   │   cain.py - main model
|   |   cain_noca.py - model without channel attention
|   |   cain_encdec.py - model with additional encoder-decoder
└───data - implements dataloaders for each dataset
│   |   vimeo90k.py - main training / testing dataset
|   |   video.py - custom data for testing
│   └───symbolic links to each dataset
|       | ...
```

## Dependencies

Current version is tested on:

- Ubuntu 18.04
- Python==3.7.5
- numpy==1.17
- [PyTorch](http://pytorch.org/)==1.3.1, torchvision==0.4.2, cudatoolkit==10.1
- tensorboard==2.0.0 (If you want training logs)
- opencv==3.4.2
- tqdm==4.39.0

``` text
# Easy installation (using Anaconda environment)
conda create -n cain
conda activate cain
conda install python=3.7
conda install pip numpy
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install tqdm opencv
```

## Usage

#### Training / Testing with Vimeo90K dataset
- First make symbolic links in `data/` folder : `ln -s /path/to/vimeo_triplet_data/ ./data/vimeo_triplet`
  - [Vimeo90K dataset](http://toflow.csail.mit.edu/)
- For training: `CUDA_VISIBLE_DEVICES=0 python main.py --exp_name EXPNAME --batch_size 16 --test_batch_size 16 --dataset vimeo90k --model cain --loss 1*L1 --max_epoch 200 --lr 0.0002`
- Or, just run `./run.sh`
- For testing performance on Vimeo90K dataset, just add `--mode test` option

#### Interpolating with custom video
- Download pretrained models from [Here](https://drive.google.com/open?id=1BHy1gkejHxy-7vCwKczTb4Jviks8KOd3)
- Prepare frame sequences in `data/frame_seq`
- run `test_custom.sh`

## Results

<center><img src="./figures/qualitative_vimeo.png" width="100%"></center>

## Citation

If you find this code useful for your research, please consider citing the following paper:

``` text
@inproceedings{choi2020cain,
    author = {Choi, Myungsub and Kim, Heewon and Han, Bohyung and Xu, Ning and Lee, Kyoung Mu},
    title = {Channel Attention Is All You Need for Video Frame Interpolation},
    booktitle = {AAAI},
    year = {2020}
}
```

## Acknowledgement

Many parts of this code is adapted from:

- [EDSR-Pytorch](https://github.com/thstkdgus35/EDSR-PyTorch)
- [RCAN](https://github.com/yulunzhang/RCAN)

We thank the authors for sharing codes for their great works.