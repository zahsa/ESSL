# FROM https://github.com/DeepVoltaire/AutoAugment/blob/master/ops.py
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from typing import List, Optional
import math

DEFAULT_OPS = {
    "HorizontalFlip":[0.0, 1.0],
    "VerticalFlip":[0.0, 1.0],
    "ShearX":[0.0, 0.3],
    "ShearY":[0.0, 0.3],
    "TranslateX":[0, int(150 / 331.0 * 32)],
    "TranslateY":[0, int(150 / 331.0 * 32)],
    "Rotate":[-30, 30],
    "Color":[0.1, 1.9],
    # "Posterize":[4, 8],
    "Solarize":[0.0, 1.0],
    "Contrast":[0.1,1.9],
    "Sharpness":[0.1, 1.9],
    "Brightness":[0.1, 1.9]
}
def HorizontalFlip(intensity: float):
    return lambda img: RandomHorizontalFlip(intensity)(img)
def VerticalFlip(intensity: float):
    return lambda img: RandomVerticalFlip(intensity)(img)
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

