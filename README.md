[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

## Rethinking Mean Teacher in Source Free Object Detection from A Prototypes Alignment Perspective

#### Contributions

- we propose class prototypes and introduce them into SFOD;
- based on the class prototypes, we propose a new SFOD method, called Class Prototypes Alignment (CPA). Prototype contrastive and prototype distillation losses are introduced to improve the model representation ability;
- extensive experiments are carried out on several experimental setups and compared with the latest methods, our method has achieved state-of-the-art (SOTA) performance.

## Contents

1. [Installation Instructions](#installation-instructions)
2. [Dataset Preparation](#dataset-preparation)
3. [Execution Instructions](#execution-instructions)
   - [Training](#training)
   - [Evaluation](#evaluation)
4. [Results](#results)

## Installation Instructions

- We use Python 3.6, PyTorch 1.9.0 (CUDA 10.2 build), reference to IRG (CVPR 2023)).
- The codebase is built on [Detectron](https://github.com/facebookresearch/detectron2).

```angular2
conda create -n cpa_sfda python=3.6

Conda activate cpa_sfda

conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

cd cpa
pip install -r requirements.txt

## Make sure you have GCC and G++ version <=8.0
cd ..
python -m pip install -e cpa

```

## Dataset Preparation

* **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets.
* **Clipart, WaterColor**: Dataset preparation instruction link [Cross Domain Detection ](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets). Images translated by Cyclegan are available in the website.
* **Sim10k**: Website [Sim10k](https://fcav.engin.umich.edu/sim-dataset/)
* **CitysScape, FoggyCityscape**: Download website [Cityscape](https://www.cityscapes-dataset.com/), see dataset preparation code in [DA-Faster RCNN](https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch)

Download all the dataset into "./dataset" folder.
The codes are written to fit for the format of PASCAL_VOC.
For example, the dataset [Sim10k](https://fcav.engin.umich.edu/sim-dataset/) is stored as follows.

```
$ cd ./dataset/Sim10k/VOC2012/
$ ls
Annotations  ImageSets  JPEGImages
$ cat ImageSets/Main/val.txt
3384827.jpg
3384828.jpg
3384829.jpg
.
.
```

## Execution Instructions

### Training

- Download the source-trained model weights in source_model folder [Link](https://drive.google.com/drive/folders/1Aia6wCHPCHGsVk8yQtuByxEyoYm1KfQq?usp=sharing)

```angular2
CUDA_VISIBLE_DEVICES=$GPU_ID python tools/train_st_sfda_net.py \ 
--config-file configs/sfda/sfda_foggy.yaml --model-dir ./source_model/cityscape_baseline/model_final.pth
```

### Evaluation

- After training, load the teacher model weights and perform evaluation using

```angular2
CUDA_VISIBLE_DEVICES=$GPU_ID python tools/plain_test_net.py --eval-only \ 
--config-file configs/sfda/foggy_baseline.yaml --model-dir $PATH TO CHECKPOINT
```

## Results

- Pre-trained models can be downloaded from [Link](https://drive.google.com/drive/folders/1RJzz4u9WV8mrcAdz_Z7_k-SSQ7SPO9hE?usp=share_link).

## Acknowledgement

We thank the developers and authors of [Detectron](https://github.com/facebookresearch/detectron2) for releasing their helpful codebases. And we thank IRG (CVPR 2023) for contributing to the open source community.
