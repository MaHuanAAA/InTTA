# Adapted from https://github.com/YBZh/Bridging_UDA_SSL

import torch
from PIL import Image, ImageOps, ImageEnhance, ImageDraw


def AutoContrast(img, _):
    return ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return ImageOps.equalize(img)


def Invert(img, _):
    return ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    return img.rotate(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    if v < 0:
        return img
    w, h = img.size
    x_center = _sample_uniform(0, w)
    y_center = _sample_uniform(0, h)

    x0 = int(max(0, x_center - v / 2.0))
    y0 = int(max(0, y_center - v / 2.0))
    x1 = min(w, x0 + v) 
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img



from PIL import Image
import numpy as np
import random

def Randpatch(img, v):
    assert v >= 0.0
    img_np = np.array(img)
    height, width, _ = img_np.shape
    patch_size = int(width/8)

    for _ in range(int(v)):
        # 随机选择源和目标区域的左上角坐标
        src_x = random.randint(0, width - patch_size)
        src_y = random.randint(0, height - patch_size)
        dst_x = random.randint(0, width - patch_size)
        dst_y = random.randint(0, height - patch_size)

        # 提取源区域的patch
        src_patch = img_np[src_y:src_y + patch_size, src_x:src_x + patch_size]

        # 在目标区域粘贴patch
        img_np[dst_y:dst_y + patch_size, dst_x:dst_x + patch_size] = src_patch

    patch_size = int(width/28)

    for _ in range(int(v)):
        # 随机选择源和目标区域的左上角坐标
        src_x = random.randint(0, width - patch_size)
        src_y = random.randint(0, height - patch_size)
        dst_x = random.randint(0, width - patch_size)
        dst_y = random.randint(0, height - patch_size)

        # 提取源区域的patch
        src_patch = img_np[src_y:src_y + patch_size, src_x:src_x + patch_size]

        # 在目标区域粘贴patch
        img_np[dst_y:dst_y + patch_size, dst_x:dst_x + patch_size] = src_patch

    return Image.fromarray(img_np)


FIX_MATCH_AUGMENTATION_POOL = [
    # (AutoContrast, 0, 1),
    # (Brightness, 0.05, 0.95),
    # (Color, 0.05, 0.95),
    # (Contrast, 0.05, 0.95),
    # (Equalize, 0, 1),
    # (Identity, 0, 1),
    # (Posterize, 4, 8),
    (Rotate, -30, 30),
    # (Sharpness, 0.05, 0.95),
    (ShearX, -0.3, 0.3),
    (ShearY, -0.3, 0.3),
    # (Solarize, 0, 256),
    (TranslateX, -0.3, 0.3),
    (TranslateY, -0.3, 0.3),
]

FIX_MATCH_AUGMENTATION_POOL_strong = [
    (Randpatch, 100, 500),
]

def _sample_uniform(a, b):
    return torch.empty(1).uniform_(a, b).item()


class RandAugment:
    def __init__(self, n, augmentation_pool=FIX_MATCH_AUGMENTATION_POOL, strong=False):
        assert n >= 1, "RandAugment N has to be a value greater than or equal to 1."
        self.n = n
        self.augmentation_pool = augmentation_pool
        if strong:
            self.augmentation_pool = FIX_MATCH_AUGMENTATION_POOL_strong

    def __call__(self, img):
        ops = [
            self.augmentation_pool[torch.randint(len(self.augmentation_pool), (1,))]
            for _ in range(self.n)
        ]
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * _sample_uniform(0, 1)
            img = op(img, val)
        cutout_val = _sample_uniform(0, 1) * 0.5
        img = Cutout(img, cutout_val)
        return img
