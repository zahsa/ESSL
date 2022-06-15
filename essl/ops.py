# FROM https://github.com/DeepVoltaire/AutoAugment/blob/master/ops.py
from PIL import Image, ImageEnhance, ImageOps
from torchvision.transforms import RandomAffine
from torchvision.transforms.functional import (posterize,
                                                solarize,
                                                adjust_contrast,
                                               adjust_sharpness)
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from functools import partial
from typing import List, Tuple, Optional, Dict
import random
import math

DEFAULT_OPS = {
    "ShearX":[0.0, 0.3],
    "ShearY":[0.0, 0.3],
    "TranslateX":[0, int(150 / 331.0 * 32)],
    "TranslateY":[0, int(150 / 331.0 * 32)],
    "Rotate":[-30, 30],
    "Color":[0.1, 1.9],
    # "Posterize":[4, 8],
    "Solarize":[0, 1],
    "Contrast":[0.1,1.9],
    "Sharpness":[0.1, 1.9],
    "Brightness":[0.1, 1.9]
}

def ShearX(intensity: float,
           interpolation: InterpolationMode=InterpolationMode.NEAREST,
           fill: Optional[List[float]] = None):
    return lambda img: F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(intensity)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )

def ShearY(intensity: float,
           interpolation: InterpolationMode=InterpolationMode.NEAREST,
           fill: Optional[List[float]] = None):
    return lambda img:  F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(intensity))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )

def TranslateX(intensity: int,
               interpolation: InterpolationMode=InterpolationMode.NEAREST,
               fill: Optional[List[float]] = None):
    return lambda img: F.affine(
            img,
            angle=0.0,
            translate=[intensity, 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )

def TranslateY(intensity: int,
               interpolation: InterpolationMode=InterpolationMode.NEAREST,
               fill: Optional[List[float]] = None):
    return lambda img: F.affine(
            img,
            angle=0.0,
            translate=[0, intensity],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )

def Rotate(intensity: int,
           interpolation: InterpolationMode=InterpolationMode.NEAREST,
           fill: Optional[List[float]] = None,):
    return lambda img: F.rotate(img, intensity, interpolation=interpolation, fill=fill)

def Brightness(intensity: float):
    return lambda img: F.adjust_brightness(img, 1.0 + intensity)

def Color(intensity: float):
    return lambda img: F.adjust_saturation(img, 1.0 + intensity)
def Posterize(intensity: int):
    return lambda img:  F.posterize(img, intensity)

def Solarize(intensity:float):
    return lambda img: F.solarize(img, intensity)

def Contrast(intensity: float):
    return lambda img: F.adjust_contrast(img, 1.0 + intensity)

def Sharpness(intensity: float):
    return lambda img: F.adjust_sharpness(img, 1.0 + intensity)

def Brightness(intensity: float):
    return lambda img: F.adjust_brightness(img, 1.0 + intensity)

def AutoContrast():
    return lambda img: F.autocontrast(img)

def Equalize():
    return lambda img:  F.equalize(img)

def Invert():
    return lambda img: F.invert(img)

def Identity():
    return lambda img: img







# class ShearX(object):
#     def __init__(self, fillcolor=(128, 128, 128)):
#         self.fillcolor = fillcolor
#         self.range = [-0.3, 0.3]
#
#     def __call__(self, x, magnitude):
#         return x.transform(
#             x.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
#             Image.BICUBIC, fillcolor=self.fillcolor)
#
#
# class ShearY(object):
#     def __init__(self, fillcolor=(128, 128, 128)):
#         self.fillcolor = fillcolor
#
#     def __call__(self, x, magnitude):
#         return x.transform(
#             x.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
#             Image.BICUBIC, fillcolor=self.fillcolor)
#
#
# class TranslateX(object):
#     def __init__(self, fillcolor=(128, 128, 128)):
#         self.fillcolor = fillcolor
#         self.range = [-150, 150]
#
#     def __call__(self, x, magnitude):
#         return x.transform(
#             x.size, Image.AFFINE, (1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
#             fillcolor=self.fillcolor)
#
#
# class TranslateY(object):
#     def __init__(self, fillcolor=(128, 128, 128)):
#         self.fillcolor = fillcolor
#
#     def __call__(self, x, magnitude):
#         return x.transform(
#             x.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
#             fillcolor=self.fillcolor)
#
#
# class Rotate(object):
#     # from https://stackoverflow.com/questions/
#     # 5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
#     def __init__(self):
#         self.range = [-30, 30]
#     def __call__(self, x, magnitude):
#         rot = x.convert("RGBA").rotate(magnitude * random.choice([-1, 1]))
#         return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(x.mode)
#
#
# class Color(object):
#     def __init__(self):
#         self.range = [0.1, 1.9]
#     def __call__(self, x, magnitude):
#         return ImageEnhance.Color(x).enhance(1 + magnitude * random.choice([-1, 1]))
#
#
# class Posterize(object):
#     def __init__(self):
#         self.range = [4, 8]
#     def __call__(self, x, magnitude):
#         return ImageOps.posterize(x, magnitude)
#
#
# class Solarize(object):
#     def __init__(self):
#         self.range = [0, 256]
#     def __call__(self, x, magnitude):
#         return ImageOps.solarize(x, magnitude)
#
#
# class Contrast(object):
#     def __init__(self):
#         self.range = [0.1,1.9]
#     def __call__(self, x, magnitude):
#         return ImageEnhance.Contrast(x).enhance(1 + magnitude * random.choice([-1, 1]))
#
#
# class Sharpness(object):
#     def __init__(self):
#         self.range = [0.1, 1.9]
#     def __call__(self, x, magnitude):
#         return ImageEnhance.Sharpness(x).enhance(1 + magnitude * random.choice([-1, 1]))
#
#
# class Brightness(object):
#     def __init__(self):
#         self.range = [0.1, 1.9]
#     def __call__(self, x, magnitude):
#         return ImageEnhance.Brightness(x).enhance(1 + magnitude * random.choice([-1, 1]))
#
#
# class AutoContrast(object):
#     def __call__(self, x, magnitude):
#         return ImageOps.autocontrast(x)
#
#
# class Equalize(object):
#     def __call__(self, x, magnitude):
#         return ImageOps.equalize(x)
#
#
# class Invert(object):
#     def __call__(self, x, magnitude):
#         return ImageOps.invert(x)