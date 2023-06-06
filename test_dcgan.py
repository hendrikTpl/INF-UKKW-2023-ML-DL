# Kelompok 3
# 412020001 - Nico Sanjaya
# 412020008 - Cristha Patrisya Pentury
# 412020009 - Yohanes Stefanus

import torch
import torchvision.transforms as transforms
from core_model.part_2_model import DCGAN_v2_G

if __name__ == '__main__':
    # Mendeteksi perangkat yang akan digunakan untuk komputasi (CPU atau GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NOISE_DIM = 100
    CHANNELS_IMG = 3
    FEATURES_GEN = 64

    # Membuat instance generator model DCGAN dan memindahkannya ke device yang sesuai (GPU jika tersedia, jika tidak CPU)
    gen = DCGAN_v2_G(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    # Memuat parameter (weights) yang telah dilatih pada model generator
    gen.load_state_dict(torch.load("train/DCGAN/train_dcgan.pth", map_location=device))
    # Set model ke mode evaluasi
    gen.eval()

    # Jumlah gambar yang akan digenerate
    num_images = 10
    # Noise input tetap untuk digunakan dalam pembuatan gambar-gambar tersebut
    fixed_noise = torch.randn(num_images, NOISE_DIM, 1, 1).to(device)

    # Menghasilkan gambar-gambar palsu dengan menggunakan generator dan noise input tetap
    with torch.no_grad():
        fake = gen(fixed_noise).detach().cpu()

    # Menyimpan gambar-gambar yang dihasilkan
    for i in range(num_images):
        img = transforms.ToPILImage()(fake[i])
        img.save(f"test/DCGAN/generated_image_{i}.png")

    print("Images generated successfully.")
