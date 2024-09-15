# TransAdapter: Vision Transformer for Feature-Centric Unsupervised Domain Adaptation

Official PyTorch implementation of [**TransAdapter: Vision Transformer for Feature-Centric Unsupervised Domain Adaptation**]().

[![Star on GitHub](https://img.shields.io/github/stars/enesdoruk/TransAdapter.svg?style=social)](https://github.com/enesdoruk/TransAdapter)


**Authors:** [Abdullah Enes Doruk](www.linkedin.com/in/enesdrk) and [Hasan F. Ates](https://www.linkedin.com/in/hasan-f-ates-9a5b6812/)


Our feature-centric unsupervised domain adaptive Swin transformer.

<p align="center">
<img src="figures/arch.png" width=90% height=70% 
class="center">
</p>

## Abstract
In this work, we propose a novel UDA approach based on the Swin Transformer, introducing three key modules to improve domain adaptation. First, we develop a Graph Domain Discriminator that plays a crucial role in domain alignment by capturing pixel-wise correlations through a graph convolutional layer, operating on both shallow and deep features. This module also calculates entropy for the query and key attention outputs to better distinguish between the source and target domains. In particular, our model does not include a task-specific domain alignment module, making it more versatile for various applications. Second, we present an Adaptive Double Attention module that simultaneously processes windows and shifted windows attention to increase long-range dependency features. An attention reweighting mechanism is employed to dynamically adjust the contributions of these attentions, thereby improving feature alignment between domains. Finally, we introduce Cross-Feature Transform, where random Swin Transformer blocks are selectively transformed using our proposed transform module, enhancing the model’s ability to generalize across domains by transferring the source style to the target. Extensive experiments demonstrate that our method achieves state-of-the-art performance on several challenging UDA benchmarks, confirming the effectiveness of our approach.

## Installation (Python 3.8.19)

This project tested under pytorch 2.4.1 and CUDA 12.4 versions. However, you can work with CUDA 11x and related Pytorch versions. 

**a.** Create environment
```Shell
conda env create -f environment.yml
```

**b.** Activate environment
```Shell
conda activate dom
```

Install fused window process for acceleration, activated by passing --fused_window_process in the running script

```Shell
cd kernels/window_process
python setup.py install #--user
```

## Pretrained Swin
Download the following models and put them in checkpoints/

* Swin-B [(ImageNet)](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth)


## Datasets
- Download [data](https://drive.google.com/file/d/1rnU49vEEdtc3EYVo7QydWzxcSuYqZbUB/view?usp=sharing) and replace the current `data/`

- Download images from [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw), [VisDA-2017](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) and put them under `data/`. For example, images of `Office-31` should be located at `data/office/domain_adaptation_images/`

## Training Baseline (For Pseudo Label Creation)
Command can be found training bash file  `scripts/training_swin.sh`.

```
python train_swin.py --train_batch_size 64 --dataset office \
        --source_list data/office/webcam_list.txt \ 
        --target_list data/office/amazon_list.txt \
        --test_list_source data/office/webcam_list.txt \
        --test_list_target data/office/amazon_list.txt \
        --pretrained_dir checkpoint/swin_base_patch4_window7_224_22k.pth \ 
        --num_steps 5000 --num_classes 31 
```

## Training TransAdapter
Command can be found training bash file  `scripts/training_transadapter.sh`.
If you do not want to train baseline model, you can set ```--pseudo_lab False```

```
python train_transadapter.py --train_batch_size 64 --dataset office \
        --source_list data/office/webcam_list.txt \ 
        --target_list data/office/amazon_list.txt \
        --test_list_source data/office/webcam_list.txt \
        --test_list_target data/office/amazon_list.txt \
        --num_classes 31 \
        --pretrained_dir checkpoint/swin_base_patch4_window7_224_22k.pth \ 
        --num_steps 15000 --gamma 0.1 --beta 0.01 -thata 0.0001 \
        --pseudo_lab True 
```

## Benchmark Results
Office-31 Dataset Results
<p align="center">
<img src="figures/res.png" width=90% height=70% 
class="center">
</p>

## Acknowledgement
- We thank the authors of [TVT](https://github.com/uta-smile/TVT/tree/main) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer/tree/main) and  their open-source codes.

## Citation
