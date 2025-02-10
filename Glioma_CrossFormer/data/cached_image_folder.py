import io
import os
import time
import torch.distributed as dist
import torch.utils.data as data
from PIL import Image
import numpy as np
from .zipreader import is_zip_path, ZipReader
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def make_dataset_with_ann(ann_file, img_prefix, extensions):
    images = []
    with open(ann_file, "r") as f:
        contents = f.readlines()
        for line_str in contents:
            path_contents = [c for c in line_str.split('\t')]
            im_file_name = path_contents[0]
            class_index = int(path_contents[1])

            assert str.lower(os.path.splitext(im_file_name)[-1]) in extensions
            item = (os.path.join(img_prefix, im_file_name), class_index)

            images.append(item)

    return images





def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if isinstance(path, bytes):
        img = Image.open(io.BytesIO(path))
    elif is_zip_path(path):
        data = ZipReader.read(path)
        img = Image.open(io.BytesIO(data))
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
    return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_img_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)





class CustomDataset(data.Dataset):
    def __init__(self, data_dir, label_file,is_train=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.is_train=is_train
        # 读取标签文件
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                filename, label = line.strip().split('\t')
                # print(filename, label)
                self.samples.append((os.path.join(data_dir, filename), int(label)))
            # print('samples',len(self.samples))

    def __getitem__(self, index):


        path, target = self.samples[index]
            # 尝试使用 NumPy 读取
        try:
            data = np.load(path)
            data=np.asarray(data, np.float32).transpose(2,1,0)
            # data = torch.from_numpy(data)
            # print('data',data.shape)
            data=data
        except Exception as e:
            raise RuntimeError(f"读取 {path} 失败: {e}")

        # # 如果有变换，应用变换
        # if self.transform is not None:
        #     print('data1',data.shape)
        #     # data = Image.fromarray(data.transpose(2,1,0))
        #     # data=torch.from_numpy(data)
        #     ToTensor=transforms.ToTensor()
        #     # data=ToTensor(data)
        #     data = self.transform(data)
        #     # data = np.asarray(data).transpose(2,1,0)
        #     print('data2',data.shape)
        # 应用变换
        # print('data1',data.shape)
        if self.transform is not None:
            if self.is_train :
                transformed_channels = []
                # 拆分6通道数据6*224*224
                # channels = [data[i:(i+1)*3, :, :] for i in range(data.shape[0])]
                data=data.transpose(2,1,0)
                channels = [data[:,:,0:3],data[:,:,3:6]]
                for channel in channels:
                    # 将单通道数据转换为PIL图像
                    # print('channel',channel.shape)
                    # channel_image = transforms.ToPILImage()(channel)
                    channel_image=Image.fromarray(channel,mode='RGB')
                    # print('channel',channel.type)

                    # channel_image = transforms.ToPILImage()(channel)
                    # 应用数据增强
                    transformed_channel = self.transform(channel_image)
                    # 将PIL图像转换回Tensor
                    # transformed_channel = transforms.ToTensor()(transformed_channel)
                    transformed_channel =np.asarray(transformed_channel)
                    transformed_channels.append(transformed_channel)
                data=data.transpose(2,1,0)
                # 重组通道
                # data = np.stack(transformed_channels, axis=2)
                data = np.concatenate((transformed_channels[0],transformed_channels[1]) ,axis=0)
                # print('data2',data.shape)
            else: #验证集模拟训练集的数据变换
                transformed_channels = []
                # 拆分6通道数据6*224*224
                # channels = [data[i:(i+1)*3, :, :] for i in range(data.shape[0])]
                data=data.transpose(2,1,0)
                channels = [data[:,:,0:3],data[:,:,3:6]]
                for channel in channels:
                    transformed_channels.append(transformed_channel)
                data=data.transpose(2,1,0)
                # 重组通道
                data = np.concatenate((transformed_channels[0],transformed_channels[1]) ,axis=0)

        return data, target

    def __len__(self):
        return len(self.samples)
