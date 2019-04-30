import torch
import torch.utils.data as data

import numpy as np
import math
import random
import os
from os import listdir

from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])

def default_loader(path):
    return Image.open(path).convert('RGB')

def toTensor(pic):
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img.float().div(255)

    # PIL image.
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I:16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

class CelebA(data.Dataset):
    def __init__(self, data_path='data/', load_size=64, fine_size=64, flip=1):
        super(CelebA, self).__init__()
        self.image_list = [x for x in listdir(data_path) if is_image_file(x)]
        self.data_path = data_path
        self.load_size = load_size
        self.fine_size = fine_size
        self.flip = flip

    def __getitem__(self, item):
        path = os.path.join(self.data_path, self.image_list[item])
        img = default_loader(path)
        w, h = img.size

        if h != self.load_size:
            img = img.resize((self.load_size, self.load_size), Image.BILINEAR)

        if self.load_size != self.fine_size:
            x1 = math.floor((self.load_size - self.fine_size)/2)
            y1 = math.floor((self.load_size - self.fine_size)/2)
            img = img.crop((x1, y1, x1+self.fine_size, y1+self.fine_size))

        if self.flip == 1:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = toTensor(img)     # 3 * width * height.
        img = img.mul_(2).add_(-1)
        return img

    def __len__(self):
        return len(self.image_list)

def get_loader(root, batch_size, scale_size, num_workers=12, shuffle=True):
    dataset = CelebA(root, scale_size, scale_size, 1)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)
    return data_loader