# /**
#  * @author Hendrik
#  * @email hendrik.gian@gmail.com
#  * @create date 2023-03-16 13:42:16
#  * @modify date 2023-03-16 13:42:16
#  * @desc [description]
#  */

import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T


both_transform = A.Compose(
    [A.Resize(width=512, height=512),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                    0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                    0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

# TODO: sesuaikan dataset UKRIDA dataset, indonesian food street, 3 label, bakso, mie ayam, batagor
# indonesian street food

# non-aligned data 256x256x3


class UkridaDataset_v1(Dataset):
    pass
    # folder base data loader
    # metadata.csv data loader


# Pix2Pix Dataloader from folder
# A, B AB -> aligned data
class UkridaDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        # make sure all the listed files are in FILE_EXT
        self.list_files = [i for i in self.list_files if i.endswith(".png")]
        # print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        w_img = image.shape[1]  # get the width size of image
        w_img_half = w_img // 2  # assuming dataset img is aligned x,y
        # since input and target images were aligned side by side, divided into two 512X512
        input_image = image[:, :w_img_half, :]
        target_image = image[:, w_img_half:, :]
        # input_image = image[:, :512, :]
        # target_image = image[:, 512:, :]

        augmentations = both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = transform_only_input(image=input_image)["image"]
        target_image = transform_only_mask(image=target_image)["image"]

        return input_image, target_image


# test case
if __name__ == "__main__":
    dataset = UkridaDataset("./data/indonesian-street-food/AB/train")
    loader = DataLoader(dataset)
    for x, y in loader:
        print(x.shape)
        # save_image(x, "x.png")
        # save_image(y, "y.png")
        import sys
        sys.exit()
