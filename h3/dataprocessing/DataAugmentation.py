from torchvision import transforms
import torch.nn as nn
import torch
import random


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
                 noise_amount: int = 20,   # the std of the noise level, assuming the image pixel value is between 0 to 255
                 flip_probability: float = 0.5,
                 scale_range: tuple[float, float] = (0.9, 1.1),
                 solarize_threshold: int = 128,    # above the solarize_threshold, the pixle will be randomly inverted with probability to 'solarize_probability'
                 solarize_probability: float = 0.5,
                 cj_brightness=(0.95,1), # colorjitter brightness
                 cj_contrast=(0.95,1),  # colorjitter contrast
                 cj_saturation=(0.95,1),  # colorjitter saturation
                 cj_hue=(-0.15,0.15)) -> None:  ## colorjitter hue
        super().__init__()

        #the following code apply gaussian noise to the image with the std value of the noise specified by 'noise_amount'
        self.apply_noise = transforms.Compose([
                            transforms.Lambda(lambda x: x + torch.randn(x.size())*noise_amount),
                            transforms.Lambda(lambda x: torch.clamp(x, 0, 255)),
                            transforms.Lambda(lambda x: x.int())
        ])


        self.flipping = nn.Sequential(
            transforms.RandomHorizontalFlip(p=flip_probability),
            transforms.RandomVerticalFlip(p=flip_probability)
        )

        self.rotation = transforms.Lambda(lambda x: transforms.functional.rotate(x, random.choice([0, 90, 180, 270])))
            

        #the following code apply zoom 
        self.zoom = transforms.RandomAffine(degrees=(0, 0), translate=(0, 0), scale=scale_range)
        #the following code apply solarize
        #please set the threshold(range:0 to 255), which when exceeded the pixel will be solarized.
        #let the threhold to be 0.8 to 1 times of the brightest pixel. 
        #if set threhold very low (like 0.1) the effect will be like invert, which differs from original a lot.
        self.solarize = transforms.RandomSolarize(solarize_threshold,solarize_probability)

        # the following code apply colorjitte to the image
        self.colorjitter=transforms.Compose([
                            transforms.Lambda(lambda x: x/255),
                            transforms.ColorJitter(
                                brightness=(0.95, 1),
                                contrast=(0.95, 1),
                                saturation=(0.95, 1),
                                hue=(-0.15, 0.15)
                            ),
                            transforms.Lambda(lambda x: x*255),
                            transforms.Lambda(lambda x: x.int())
        ])
                
        self.use_noise = use_noise
        self.use_flipping = use_flipping
        self.use_rotation = use_rotation
        self.use_zoom = use_zoom
        self.use_solarize = use_solarize
        self.use_colorjitter = use_colorjitter

    @torch.no_grad()  # disable gradients for effiency
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
