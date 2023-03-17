from torchvision import transforms
import pandas as pd
from torchvision.models import ViT_L_16_Weights
from torchvision.models import Swin_V2_B_Weights
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from h3.models.opti_utils import load_full_ram, open_img
from h3 import logger


class HurricaneDataset(Dataset):
    def __init__(
            self,
            dataframe,
            img_path,
            EF_features,
            image_embedding_architecture,
            augmentations=None,
            zoom_levels=None,
            ram_load: bool = False
    ):
        self.dataframe = dataframe
        self.img_path = img_path
        self.EF_features = EF_features
        self.zoom_levels = ["1"] if zoom_levels is None else zoom_levels

        if image_embedding_architecture == "ResNet18":
            self.preprocessing = transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            
        elif image_embedding_architecture == "ViT_L_16":
            self.preprocessing = ViT_L_16_Weights.IMAGENET1K_V1.transforms()
        
        elif image_embedding_architecture == "Swin_V2_B":
            self.preprocessing = Swin_V2_B_Weights.IMAGENET1K_V1.transforms()

        elif image_embedding_architecture == "SatMAE":
            # values from CustomDatasetFromImages() in https://github.com/sustainlab-group/SatMAE/blob/main/util/datasets.py
            self.preprocessing = transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4182007312774658, 0.4214799106121063, 0.3991275727748871], 
                                         std=[0.28774282336235046, 0.27541765570640564, 0.2764017581939697]),
                ])

        if augmentations is not None:
            self.transform = transforms.Compose([augmentations, self.preprocessing])
        else:
            self.transform = self.preprocessing

        self.ram_load = ram_load

        if self.ram_load:
            logger.info("Using full RAM loading. Hold on!")
            image_id = [self.dataframe["id"].iloc[idx] for idx in range(len(self.dataframe))]
            self.lst_paths = {v: [] for v in zoom_levels}
            for zoom_level in self.zoom_levels:
                for image in image_id:
                    path = os.path.join(self.img_path, f"zoom_{zoom_level}", f"{image}.png")
                    self.lst_paths[zoom_level].append(path)

            self.all_images = {
                zoom: load_full_ram(zoom_path, transform=self.transform) for zoom, zoom_path in self.lst_paths.items()
            }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx) -> tuple:
        if self.ram_load:
            x = {}
            zoomed_images = {}

            for zoom_level in self.zoom_levels:
                zoomed_images = {"img_zoom_" + zoom_level: self.all_images[zoom_level][idx]}

            for key in self.EF_features:
                x.update(
                    {key: torch.as_tensor(self.dataframe[self.EF_features[key]].iloc[idx]).type(torch.FloatTensor)}
                )
            label = torch.as_tensor(self.dataframe["damage_class"].iloc[idx]).type(torch.LongTensor)
            x.update(zoomed_images)

        else:
            image_id = self.dataframe["id"].iloc[idx]
            x = {}
            zoomed_images = {}

            for zoom_level in self.zoom_levels:
                path = os.path.join(self.img_path, "zoom_" + zoom_level, str(image_id) + ".png")

                img = open_img(path, transform=self.transform)
                # img = Image.open(path)
                # img = self.transform(img)
                # img = np.asarray(img)
                #img = np.swapaxes(img, 0, 2)

                zoomed_images["img_zoom_" + zoom_level] = img

            #idx_EFs = [int(self.dataframe[ef].iloc[idx]) for ef in EF_features]

            # for each of the different types of EF (e.g. weather, soil, DEM) grab
            # their associated values and put them into the dictionary
            for key in self.EF_features:
                x.update({key: torch.as_tensor(self.dataframe[self.EF_features[key]].iloc[idx]).type(torch.FloatTensor)})

            # from risk df
            # storm_surge_ef = self.dataframe["max_sust_wind"].iloc[idx]

            label = torch.as_tensor(self.dataframe["damage_class"].iloc[idx]).type(torch.LongTensor)

            # add Weather EFs

            # put it in a dictionary so don't have to return a tonne of different values
            # img also goes in the below dictionary
            # x = {"storm_surge_ef": storm_surge_ef, "soil_ef": soil_ef}
            # EFs = concat all EFs
            # 0-1 normalize all EFs
            # mean, std
            # x = {"EFs": idx_EFs}
            x.update(zoomed_images)

        #print("x=",x)
        # torch.nn.CrossEntropyLoss expects integer labels, not one-hot labels
        # see https://stackoverflow.com/questions/62456558/is-one-hot-encoding-required-for-using-pytorchs-cross-entropy-loss-function
        # label = F.one_hot(label, num_classes = 5)

        return x, label
