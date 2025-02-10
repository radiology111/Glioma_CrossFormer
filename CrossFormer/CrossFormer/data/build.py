import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp
from .tfrecord_torch_loader import ImageTFRecordDataSet

from .cached_image_folder import CustomDataset#,CachedImageFolder,
from .samplers import SubsetRandomSampler


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_custom_dataset(is_train=True, config=config)
    config.freeze()
    # print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_custom_dataset(is_train=False, config=config)
    # print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    # num_tasks = dist.get_world_size()
    # # global_rank = dist.get_rank()
    # if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
    #     indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
    #     sampler_train = SubsetRandomSampler(indices)
    # else:
    #     sampler_train = torch.utils.data.DistributedSampler(
    #         dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #     )

    # indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    # sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_custom_dataset(is_train, config):
    # transform = build_transform(is_train, config)
   
    if is_train: 
        index_file=config.DATA.TRAIN_idx
    else :
        index_file=config.DATA.VAL_idx
    transform=build_transform(is_train, config)
    dataset = CustomDataset(config.DATA.DATA_PATH, index_file ,is_train,transform=transform )
    nb_classes = 2


    return dataset, nb_classes

# def build_dataset(is_train, config):
#     transform = build_transform(is_train, config)
#     if config.DATA.DATASET == 'imagenet':
#         prefix = 'train' if is_train else 'validation'
#         if config.DATA.ZIP_MODE:
#             ann_file = prefix + "_map.txt"
#             prefix = prefix + ".zip@/"
#             dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
#                                         cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
#         else:
#             root = os.path.join(config.DATA.DATA_PATH, prefix)
#             dataset = datasets.ImageFolder(root, transform=transform)
#         nb_classes = 1000
#     elif config.DATA.DATASET == 'ImageNet': ## tfrecord
#         if is_train:
#             records = [os.path.join(config.DATA.DATA_PATH, "train.tfrecord")]
#         else:
#             records = [os.path.join(config.DATA.DATA_PATH, "validation.tfrecord")]
#         dataset = ImageTFRecordDataSet(records, transform)
#         nb_classes = 1000
#     elif config.DATA.DATASET == 'ImageNet22K': ## tfrecord
#         root = config.DATA.DATA_PATH
#         records = [os.path.join(root, filename) for filename in os.listdir(root) if ".tfrecord" in filename]
#         dataset = ImageTFRecordDataSet(records, transform)
#         nb_classes = 21841
#     else:
#         raise NotImplementedError("We only support ImageNet (for tfrecord), imagenet, and ImageNet22K Now.")

#     return dataset, nb_classes


# def build_transform(is_train, config):
#     resize_im = config.DATA.IMG_SIZE > 32
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=config.DATA.IMG_SIZE,
#             is_training=True,
#             color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
#             auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
#             re_prob=config.AUG.REPROB,
#             re_mode=config.AUG.REMODE,
#             re_count=config.AUG.RECOUNT,
#             interpolation=config.DATA.INTERPOLATION,
#         )
#         if not resize_im:
#             # replace RandomResizedCropAndInterpolation with
#             # RandomCrop
#             transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
#         return transform

#     t = []
#     if resize_im:
#         if config.TEST.CROP:
#             size = int((256 / 224) * config.DATA.IMG_SIZE)
#             t.append(
#                 transforms.Resize(size, interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)),
#                 # to maintain same ratio w.r.t. 224 images
#             )
#             t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
#         else:
#             t.append(
#                 transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
#                                   interpolation=str_to_pil_interp(config.DATA.INTERPOLATION))
#             )

#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
#     return transforms.Compose(t)




def build_transform(is_train, config):
    if is_train:
        # 训练集的数据增强
        transformations = [
            transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.8, 1.0)),#随机裁剪
            #     随机裁剪图像
            # transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4),
            transforms.RandomRotation(degrees=15), #随机旋转
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomVerticalFlip(p=0.1),  # 随机垂直翻转
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色调整
            #transforms.RandomRotation(degrees=15),  # 随机旋转
            #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # 仿射变换
            #transforms.RandomPerspective(distortion_scale=0.5),  # 透视变换
            # transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random', inplace=False)  # 随机擦除
        ]

        transform = transforms.Compose([
            transforms.RandomChoice(transformations),  # 每次从上述变换中随机选择一种数据增强
            transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
 

        return transform