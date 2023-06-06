# Kelompok 3
# 412020001 - Nico Sanjaya
# 412020008 - Cristha Patrisya Pentury
# 412020009 - Yohanes Stefanus

import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

# Model Generator untuk simple GAN
class simple_G(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.generator_G = nn.Sequential(
            nn.Linear(z_dim, 256),  # Layer linier untuk menghubungkan noise ke lapisan tersembunyi
            nn.LeakyReLU(0.01),  # Fungsi aktivasi LeakyReLU untuk memperkenalkan non-linearitas
            nn.Linear(256, img_dim),  # Layer linier untuk menghasilkan gambar output
            nn.Tanh(),  # Fungsi aktivasi Tanh untuk membatasi nilai output antara -1 dan 1
        )

    def forward(self, x):
        return self.generator_G(x)  # Langkah maju dari generator

# Model Diskriminator untuk simple GAN
class simple_D(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.discriminator_D = nn.Sequential(
            nn.Linear(in_features, 128),  # Layer linier untuk menghubungkan input ke lapisan tersembunyi
            nn.LeakyReLU(0.1),  # Fungsi aktivasi LeakyReLU untuk memperkenalkan non-linearitas
            nn.Linear(128, 1),  # Layer linier untuk menghasilkan nilai probabilitas diskriminasi
            nn.Sigmoid(),  # Fungsi aktivasi sigmoid untuk menghasilkan nilai probabilitas antara 0 dan 1
        )

    def forward(self, x):
        return self.discriminator_D(x)  # Langkah maju dari diskriminator

# Model Diskriminator berdasarkan AlexNet
class Alex_D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass
    pass

    def forward(self, x):
        pass

class DCGAN_v2_D(nn.Module):
    def __init__(self, channels_img, features_d):
        super(DCGAN_v2_D, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),  # Layer konvolusi 1
            nn.LeakyReLU(0.2),  # Fungsi aktivasi LeakyReLU
            self._block(features_d, features_d * 2, 4, 2, 1),  # Blok konvolusi 1
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # Blok konvolusi 2
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # Blok konvolusi 3
            self._block(features_d * 8, features_d * 16, 4, 2, 1),  # Blok konvolusi 4 (Baru)
            nn.Conv2d(features_d * 16, 1, kernel_size=4, stride=1, padding=0),  # Convolutional layer terakhir
            nn.Sigmoid()  # Fungsi aktivasi sigmoid
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.discriminator(x)

class DCGAN_v2_G(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(DCGAN_v2_G, self).__init__()
        self.generator = nn.Sequential(
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # Blok dekonvolusi 1
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # Blok dekonvolusi 2
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # Blok dekonvolusi 3
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # Blok dekonvolusi 4
            self._block(features_g * 2, features_g, 4, 2, 1),  # Blok dekonvolusi 5 (Baru)
            nn.ConvTranspose2d(
                features_g, channels_img, kernel_size=4, stride=2, padding=1
            ),  # Layer dekonvolusi terakhir
            nn.Tanh()  # Fungsi aktivasi Tanh
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.generator(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)