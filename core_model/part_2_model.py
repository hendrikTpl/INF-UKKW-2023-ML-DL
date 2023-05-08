# /**
#  * @author Hendrik
#  * @email [hendrik.gian@gmail.com]
#  * @create date 2023-04-17 23:31:58
#  * @modify date 2023-04-17 23:31:58
#  * @desc [description]
#  */

# A) simpleGAN
# B) Deep Conv Net GAN (DCGAN)implementation from the scratch

# import libraries yang dibutuhkan
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import matplotlib.pyplot as plt
import numpy as np


# ini implementasi GAN sederhana
# discriminator D dan generator G yang digunakan disini adalah Fully-Connected NN
class simple_G(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.generator_G = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.generator_G(x)


class simple_D(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.discriminator_D = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.discriminator_D(x)


"""
Impementasi Deep Conv Net GAN (DCGAN)

"""

# class Discriminator


class DCGAN_D(nn.Module):
    def __init__(self, channels_img, features_d):
        super(DCGAN_D, self).__init__()
        self.discriminator_D = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.discriminator_D(x)

# Generator G


class DCGAN_G(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(DCGAN_G, self).__init__()
        self.generator_G = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.generator_G(x)

# inisialisasi weight


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

#
# TODO:
# supposedly you want to add new model, You can also define your customize  generator and discriminator GAN
# DCGAN with custom dataset -> indonesian street food dataset
# make your custom dataloader
