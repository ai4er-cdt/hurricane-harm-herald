# --------------------------------------------------------
# References:
# SatMAE: https://github.com/sustainlab-group/SatMAE
# SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery
# Yezhen Cong and Samar Khanna and Chenlin Meng and Patrick Liu and Erik Rozi and Yutong He and Marshall Burke and David B. Lobell and Stefano Ermon
# CC BY-NC 4.0 License https://github.com/sustainlab-group/SatMAE/blob/main/LICENSE
#
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from h3.utils.downloader import downloader
from h3.utils.directories import get_download_dir
import torch
import torch.nn as nn
import os

# fork SatMAE repo, edit code and remove useless bits
# make it pip installable (look at SatMAe env.yaml)
# use os.join instead of + (wuindows compatability)
# use downloader(), not ur_dowload()

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def download_model() -> None:
    if not os.path.exists(os.path.join(get_download_dir(), "fmow_pretrain.pth")):
        downloader(["https://zenodo.org/record/7369797/files/fmow_pretrain.pth"], get_download_dir())

def get_model():
    from h3.models.SatMAE.models_vit import vit_large_patch16
    from h3.models.SatMAE.pos_embed import interpolate_pos_embed

    # download the model if not already downloaded
    download_model()

    # the rest of this code comes from https://github.com/sustainlab-group/SatMAE/blob/main/main_finetune.py

    # init the ViT_L model
    model = vit_large_patch16(
        patch_size=16, img_size=224, in_chans=3,
        num_classes=1000, drop_path_rate=0.0, global_pool=False,
    )

    # load in SatMAE weights
    checkpoint = torch.load(os.path.join(get_download_dir(), "fmow_pretrain.pth"), map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # put weights into model
    model.load_state_dict(checkpoint_model, strict=False)

    # replace classification layer with identity function
    model.head = Identity()

    return model