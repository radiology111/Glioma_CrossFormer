# CrossFormer++

This repository is the code for our papers:

- [CrossFormer++: A Versatile Vision Transformer Hinging on Cross-scale Attention](https://arxiv.org/abs/2303.06908) (**IEEE TPAMI Acceptance**).

  - Authors: [Wenxiao Wang](https://www.wenxiaowang.com), Wei Chen, Qibo Qiu, [Long Chen](https://zjuchenlong.github.io/), Boxi Wu, Binbin Lin, [Xiaofei He](http://www.cad.zju.edu.cn/home/xiaofeihe/), Wei Liu
 
- [CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention](https://arxiv.org/pdf/2108.00154.pdf) (**ICLR 2022 Acceptance**).

  - Authors: [Wenxiao Wang](https://www.wenxiaowang.com), Lu Yao, [Long Chen](https://zjuchenlong.github.io/), Binbin Lin, [Deng Cai](http://www.cad.zju.edu.cn/home/dengcai/), [Xiaofei He](http://www.cad.zju.edu.cn/home/xiaofeihe/), Wei Liu

The [crossformer](https://github.com/cheerss/CrossFormer/tree/crossformer) branch retains the old version code with old dependencies.

## Updates

- [x] Mask-RCNN detection/instance segmentation results with 3x training schedule.
- [x] Cascade Mask-RCNN detection/instance segmentation results with 3x training schedule.
- [x] The usage of `get_flops.py` in detection and segmentation.
- [x] Upload the pretrained CrossFormer-L.
- [x] Upload the pretrained models for CrossFormer++-S/B/L/H classification.
- [ ] Upload CrossFormer++-S/B/L for detection and segmentation.




## Introduction

Existing vision transformers fail to build attention among objects/features of different scales (cross-scale attention), while such ability is very important to visual tasks. **CrossFormer** is a versatile vision transformer which solves this problem. Its core designs contain **C**ross-scale **E**mbedding **L**ayer (**CEL**), **L**ong-**S**hort **D**istance **A**ttention (**L/SDA**), which work together to enable cross-scale attention.

**CEL** blends every input embedding with multiple-scale features. **L/SDA** split all embeddings into several groups, and the self-attention is only computed within each group (embeddings with the same color border belong to the same group.).

Besides, we also propose a **D**ynamic **P**osition **B**ias (**DPB**) module, which makes the effective yet inflexible relative position bias apply to variable image size.

![](./figures/github_pic.png)

Further, in CrossFormer++, we introduce a **P**rogressive **G**roup **S**ize (**PGS**) strategy to achieve a better balance between performance and computation budget and a **A**ctivation **C**ooling **L**ayer (**ACL**) to suppress the magnitude of activations that grows drastically in the residual stream.

![](./figures/github_pic_2.png)

Now, experiments are done on four representative visual tasks, *i.e.*, image classification, objection detection, and instance/semantic segmentation. Results show that CrossFormer outperforms existing vision transformers in these tasks, especially in dense prediction tasks (*i.e.*, object detection and instance/semantic segmentation). We think it is because image classification only pays attention to one object and large-scale features, while dense prediction tasks rely more on cross-scale attention.




## Prerequisites

1. Create and activate conda environment
```bash
conda create -n crossformer_env python=3.9 -y
conda activate crossformer_env
```

2. Libraries (Python3.9-based)
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install tensorboard termcolor
pip install timm pyyaml yacs protobuf==3.20.0
```

3. Dataset: ImageNet

4. Requirements for detection/instance segmentation and semantic segmentation are listed here: [detection/README.md](./detection/README.md) or [segmentation/README.md](./segmentation/README.md)

For ease of use, we have adapted our code of CrossFormer and CrossFormer++ with newer version of pytorch, mmcv, mmdetection and mmsegmentation, so the results in this repository may be slightly different from the results reported in the paper.

If you're using relatively old versions of CUDA, please consider using our original CrossFormer code at `crossformer` branch of this repository.




## Getting Started

### Training
```bash
## There should be two directories under the path_to_imagenet: train and validation

## CrossFormer-T
python -u -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/crossformer/tiny_patch4_group7_224.yaml \
--batch-size 64 --data-path path_to_imagenet --output ./output

## CrossFormer-S
python -u -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/crossformer/small_patch4_group7_224.yaml \
--batch-size 64 --data-path path_to_imagenet --output ./output

## CrossFormer-B
python -u -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/crossformer/base_patch4_group7_224.yaml 
--batch-size 64 --data-path path_to_imagenet --output ./output

## CrossFormer-L
python -u -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/crossformer/large_patch4_group7_224.yaml \
--batch-size 64 --data-path path_to_imagenet --output ./output

## CrossFormer++-S
python -u -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/crossformer_pp/small_patch4_group_const_224.yaml \
--batch-size 64 --data-path path_to_imagenet --output ./output

## CrossFormer++-B
python -u -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/crossformer_pp/base_patch4_group_const_224.yaml \
--batch-size 64 --data-path path_to_imagenet --output ./output

## CrossFormer++-L
python -u -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/crossformer_pp/large_patch4_group_const_224.yaml \
--batch-size 64 --data-path path_to_imagenet --output ./output
```

### Testing
```bash
## Take CrossFormer-T as an example of evaluating accuracy
python -u -m torch.distributed.launch --nproc_per_node 1 main.py --cfg configs/crossformer/small_patch4_group7_224.yaml \
--batch-size 64 --data-path path_to_imagenet --eval --resume path_to_crossformer-t.pth

## Take CrossFormer-T as an example of testing throughput
python -u -m torch.distributed.launch --nproc_per_node 1 main.py --cfg configs/crossformer/small_patch4_group7_224.yaml \
--batch-size 64 --data-path path_to_imagenet --throughput
```

You need to modify the `path_to_imagenet` and `path_to_crossformer-t.pth` accordingly.

Training and testing scripts for objection detection: [detection/README.md](./detection/README.md).

Training and testing scripts for semantic segmentation: [segmentation/README.md](./segmentation/README.md).




## Results

### Image Classification

Models trained on ImageNet-1K and evaluated on its validation set. The input image size is 224 x 224.

| Architectures | Params | FLOPs | Accuracy | Models |
| ------------- | ------: | -----: | --------: | :---------------- |
| ResNet-50 | 25.6M | 4.1G | 76.2% |      -        |
| RegNetY-8G | 39.0M | 8.0G | 81.7% |     -        |
| **CrossFormer-T** | **27.8M**  | **2.9G**  | **81.5%**    | [Google Drive](https://drive.google.com/file/d/1YSkU9enn-ITyrbxLH13zNcBYvWSEidfq/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/1M45YXZgVvp6Ew9DO8UhdlA), key: nkju |
| **CrossFormer-S** | **30.7M**  | **4.9G**  | **82.5%**    | [Google Drive](https://drive.google.com/file/d/1RAkigsgr33va0RZ85S2Shs2BhXYcS6U8/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/1Xf4MXfb_soCnJFBeNDmoQQ), key: fgqj |
| **CrossFormer++-S** | **23.3M**  | **4.9G**  | **83.2%**    | [BaiduCloud](https://pan.baidu.com/s/1smc_kRoogd0Ig5vfqwuGTA?pwd=crsf), key:crsf |
| **CrossFormer-B** | **52.0M**  | **9.2G**  | **83.4%**    | [Google Drive](https://drive.google.com/file/d/1bK8biVCi17nz_nkt7rBfio_kywUpllSU/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/1f5dH__UGDXb-HoOPHT5p0A), key: 7md9 |
| **CrossFormer++-B** | **52.0M**  | **9.5G**  | **84.2%**    | [BaiduCloud](https://pan.baidu.com/s/13XVP0ebtNcyRMmkl11ALYg?pwd=crsf), key:crsf |
| **CrossFormer-L** | **92.0M**  | **16.1G** | **84.0%**    | [Google Drive](https://drive.google.com/file/d/1zRWByVW_KIZ87NgaBkDIm60DAsGJErdG/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/1YJLeHy_cxLBrZLklQBCA_A), key: cc89|
| **CrossFormer++-L** | **92.0M**  | **16.6G** | **84.7%**    | [BaiduCloud](https://pan.baidu.com/s/1xtPh-ndcOxRM7fPYYxIIRg?pwd=crsf), key:crsf  |
| **CrossFormer++-H** | **96.0M**  | **21.8G** | **84.9%**    | [BaiduCloud](https://pan.baidu.com/s/1xtPh-ndcOxRM7fPYYxIIRg?pwd=crsf), key:crsf  |

More results compared with other vision transformers can be seen in the [paper](https://arxiv.org/pdf/2108.00154.pdf).

**Note**: Checkpoints of CrossFormer++ will be released as soon as possible.

### Objection Detection & Instance Segmentation

Models trained on COCO 2017. Backbones are initialized with weights pre-trained on ImageNet-1K.

| Backbone      | Detection Head | Learning Schedule | Params | FLOPs  | box AP | mask AP |
| ------------- | ----------------- | -------------------- | ------: | ------: | ------: | ------: |
| ResNet-101 | RetinaNet | 1x | 56.7M | 315.0G | 38.5 | - |
| **CrossFormer-S** | RetinaNet         | 1x                   | **40.8M**  | **282.0G** | **44.4**   | -      |
| **CrossFormer++-S** | RetinaNet         | 1x                   | **40.8M**  | **282.0G** | **45.1**   | -      |
| **CrossFormer-B** | RetinaNet         | 1x                   | **62.1M**  | **389.0G** | **46.2**   | -      |
| **CrossFormer++-B** | RetinaNet         | 1x                   | **62.2M**  | **389.0G** | **46.6**   | -      |

| Backbone      | Detection Head | Learning Schedule | Params | FLOPs  | box AP | mask AP |
| ------------- | ----------------- | -------------------- | ------: | ------: | ------: | ------: |
| ResNet-101 | Mask-RCNN | 1x | 63.2M | 336.0G | 40.4 | 36.4 |
| **CrossFormer-S** | Mask-RCNN         | 1x                   | **50.2M**  | **301.0G** | **45.4**   | **41.4** |
| **CrossFormer++-S** | Mask-RCNN        | 1x                   | **43.0M**  | **287.4G** | **46.4**   | **42.1** |
| **CrossFormer-B** | Mask-RCNN         | 1x                   | **71.5M**  | **407.9G** | **47.2**   | **42.7** |
| **CrossFormer++-B** | Mask-RCNN        | 1x                   | **71.5M**  | **408.0G** | **47.7**   | **43.2** |
<!-- | **CrossFormer-S** | Mask-RCNN         | 3x                   | **50.2M**  | **291.1G** | **48.7**   | **43.9** |
| **CrossFormer-B** | Mask-RCNN         | 3x                   | **71.5M**  | **398.1G** | **49.8**   | **44.5** |
| **CrossFormer-S** | Cascade-Mask-RCNN | 3x                   | **88.0M**  | **769.7G** | **52.2**   | **45.2** | -->

More results and pretrained models for objection detection: [detection/README.md](./detection/README.md).

### Semantic Segmentation

Models trained on ADE20K. Backbones are initialized with weights pre-trained on ImageNet-1K.

| Backbone      | Segmentation Head | Iterations | Params | FLOPs   | IOU  | MS IOU |
| ------------- | -------------------- | ----------: | ------: | -------: | ----: | ------: |
| **CrossFormer-S** | FPN                  | 80K       | **34.3M**  | **209.8G**  | **46.4** | -      |
| **CrossFormer++-S** | FPN                  | 80K       | **27.1M**  | **199.5G**  | **47.4** | -      |
| **CrossFormer-B** | FPN                  | 80K       | **55.6M**  | **320.1G**  | **48.0** | -      |
| **CrossFormer++-B** | FPN                  | 80K       | **55.6M**  | **331.1G**  | **48.6** | -      |
| **CrossFormer-L** | FPN                  | 80K       | **95.4M**  | **482.7G**  | **49.1** | -      |

| Backbone      | Segmentation Head | Iterations | Params | FLOPs   | IOU  | MS IOU |
| ------------- | -------------------- | ----------: | ------: | -------: | ----: | ------: |
| ResNet-101 | UPerNet | 160K | 86.0M | 1029.G | 44.9 | - |
| **CrossFormer-S** | UPerNet              | 160K       | **62.3M**  | **979.5G**  | **47.6** | **48.4** |
| **CrossFormer++-S** | UPerNet              | 160K       | **53.1M**  | **963.5G**  | **49.4** | **50.8** |
| **CrossFormer-B** | UPerNet              | 160K       | **83.6M**  | **1089.7G** | **49.7** | **50.6** |
| **CrossFormer++-B** | UPerNet              | 160K       | **83.7M**  | **1089.8G** | **50.7** | **51.0** |
| **CrossFormer-L** | UPerNet              | 160K       | **125.5M** | **1257.8G** | **50.4** | **51.4** |

*MS IOU means IOU with multi-scale testing.*

More results and pretrained models for semantic segmentation: [segmentation/README.md](./segmentation/README.md).




## Citing Us

```
@inproceedings{wang2021crossformer,
  title = {CrossFormer: A Versatile Vision Transformer Hinging on Cross-scale Attention},
  author = {Wenxiao Wang and Lu Yao and Long Chen and Binbin Lin and Deng Cai and Xiaofei He and Wei Liu},
  booktitle = {International Conference on Learning Representations, {ICLR}},
  url = {https://openreview.net/forum?id=_PHymLIxuI},
  year = {2022}
}

@article{wang2023crossformer++,
  author       = {Wenxiao Wang and Wei Chen and Qibo Qiu and Long Chen and Boxi Wu and Binbin Lin and Xiaofei He and Wei Liu},
  title        = {Crossformer++: A versatile vision transformer hinging on cross-scale attention},
  journal      = {{IEEE} Transactions on Pattern Analysis and Machine Intelligence, {TPAMI}},
  year         = {2023},
  doi          = {10.1109/TPAMI.2023.3341806},
}
```




## Acknowledgement

Part of the code of this repository refers to [Swin Transformer](https://github.com/microsoft/Swin-Transformer).

