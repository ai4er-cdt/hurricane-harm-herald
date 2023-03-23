from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch
import random


class TransformSatMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop = transforms.CenterCrop(224)
        self.norm = transforms.Normalize(
            mean=[0.4182007312774658, 0.4214799106121063, 0.3991275727748871],
            std=[0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
        )

    def __call__(self, x):
        x = self.crop(x)
        x = x / 255
        x = self.norm(x)
        return x


class ColourJitter(nn.Module):
    """Class for torchvision's Color Jitter transformation for h3's data.

    Attributes
    ----------
    jitter : torchvision.transformers.transforms.ColorJitter
    """
    def __init__(self):
        super().__init__()
        self.jitter = transforms.ColorJitter(
            brightness=(0.95, 1),
            contrast=(0.95, 1),
            saturation=(0.95, 1),
            hue=(-0.15, 0.15)
        )

    def __call__(self, x):
        x_div = x / 255
        x_jittered = self.jitter(x_div)
        x_mul = x_jittered * 255
        return x_mul.int()


class GaussianNoise(nn.Module):
    """Class for Gaussian Noise for pytorch.

    Attributes
    ----------
    noise_amount :float
        Amount of the standard deviation of noise to add as Gaussian Noise.
    """
    def __init__(self, noise_amount):
        super().__init__()
        self.noise_amount = noise_amount

    def __call__(self, x):
        x_noise = x + torch.randn(x.size()) * self.noise_amount
        x_clamp = torch.clamp(x_noise, 0, 255)
        return x_clamp.int()


class RotationTransform(nn.Module):
    """Class for Rotation Transform for pytorch that picks a random rotation angle.

    Attributes
    ----------
    angles :list
        List of the rotation to perform.
        The value is randomly selected.
    """
    def __init__(self, angles):
        super().__init__()
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class DataAugmentation(nn.Module):
    """Module to perform data augmentation on torch tensors."""
    """Consider using the augmentations in the below link, as they work on tensors"""
    """https://pytorch.org/vision/main/transforms.html"""

    def __init__(self,
                 use_noise: bool = False,
                 use_flipping: bool = True,
                 use_rotation: bool = True,
                 use_zoom: bool = True,
                 use_solarize: bool = False,
                 use_colorjitter: bool = False,
                 noise_amount: float = 20,   # the std of the noise level, assuming the image pixel value is between 0 to 255
                 flip_probability: float = 0.5,
                 scale_range: tuple[float, float] = (0.9, 1.1),
                 solarize_threshold: int = 128,    # above the solarize_threshold, the pixle will be randomly inverted with probability to 'solarize_probability'
                 solarize_probability: float = 0.5,
                 cj_brightness=(0.95, 1),   # colorjitter brightness
                 cj_contrast=(0.95, 1),     # colorjitter contrast
                 cj_saturation=(0.95, 1),   # colorjitter saturation
                 cj_hue=(-0.15, 0.15)) -> None:  # colorjitter hue
        super().__init__()

        self.apply_noise = GaussianNoise(noise_amount)

        self.flipping = nn.Sequential(
            transforms.RandomHorizontalFlip(p=flip_probability),
            transforms.RandomVerticalFlip(p=flip_probability)
        )

        self.rotation = RotationTransform(angles=[0, 90, 180, 270])

        self.zoom = transforms.RandomAffine(degrees=(0, 0), translate=(0, 0), scale=scale_range)
        # the following code apply solarize
        # please set the threshold(range:0 to 255), which when exceeded the pixel will be solarized.
        # let the threhold to be 0.8 to 1 times of the brightest pixel.
        # if set threhold very low (like 0.1) the effect will be like invert, which differs from original a lot.
        self.solarize = transforms.RandomSolarize(solarize_threshold, solarize_probability)

        self.colorjitter = ColourJitter()
        self.use_noise = use_noise
        self.use_flipping = use_flipping
        self.use_rotation = use_rotation
        self.use_zoom = use_zoom
        self.use_solarize = use_solarize
        self.use_colorjitter = use_colorjitter

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_flipping:
            x = self.flipping(x)

        if self.use_noise:
            x = self.apply_noise(x)

        if self.use_rotation:
            x = self.rotation(x)

        if self.use_zoom:
            x = self.zoom(x)

        if self.use_solarize:
            x = self.solarize(x)

        if self.use_colorjitter:
            x = self.colorjitter(x)

        return x
