# Kelompok 3
# 412020001 - Nico Sanjaya
# 412020008 - Cristha Patrisya Pentury
# 412020009 - Yohanes Stefanus

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from MyUtils.MyDatasets import UkridaDataset_v1
from core_model.part_2_model import DCGAN_v2_G, DCGAN_v2_D, initialize_weights
from torchsummary import summary

# Mendeteksi perangkat yang akan digunakan untuk komputasi (CPU atau GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 0.0001
BATCH_SIZE = 8
IMAGE_SIZE = 128
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 50
FEATURES_DISC = 64
FEATURES_GEN = 64

# Transformasi yang akan diterapkan pada dataset
transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
])

# Membuat objek dataset dan dataloader untuk pelatihan
dataset = UkridaDataset_v1(["data/ukrida_dataset/train/Bakso", "data/ukrida_dataset/train/Batagor", "data/ukrida_dataset/train/Mie Ayam"], transform=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Membuat objek dataset dan dataloader untuk validasi
val_dataset = UkridaDataset_v1(["data/ukrida_dataset/val/Bakso", "data/ukrida_dataset/val/Batagor", "data/ukrida_dataset/val/Mie Ayam"], transform=transforms)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Membuat objek generator dan discriminator
gen = DCGAN_v2_G(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = DCGAN_v2_D(CHANNELS_IMG, FEATURES_DISC).to(device)

# Menginisialisasi bobot pada generator dan discriminator
initialize_weights(gen)
initialize_weights(disc)

# Optimizer dan loss function
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Noise yang tetap untuk generasi gambar
fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)

# Writer untuk TensorBoard
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

# Counter untuk langkah-langkah dalam pelatihan
step = 0

# Mode pelatihan generator dan discriminator
gen.train()
disc.train()

# Loop pelatihan
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        # Mengirim data ke perangkat yang digunakan
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)

        # Menghasilkan gambar palsu dari generator
        fake = gen(noise)

        # Menghitung output dan loss pada discriminator untuk gambar asli
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

        # Menghitung output dan loss pada discriminator untuk gambar palsu
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # Menghitung total loss pada discriminator dan melakukan backpropagation
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Menghitung output dan loss pada generator
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        # Menghitung total loss pada generator dan melakukan backpropagation
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Menampilkan informasi pelatihan setiap 100 batch
        if batch_idx % 100 == 0:
            num_batches = len(dataloader)
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{num_batches} Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

            # Menyimpan gambar nyata dan palsu pada TensorBoard
            with torch.no_grad():
                real = nn.functional.interpolate(real, size=IMAGE_SIZE)
                fake = nn.functional.interpolate(fake, size=IMAGE_SIZE)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image("DCGAN Real v2", img_grid_real, global_step=step)
                writer_fake.add_image("DCGAN Fake v2", img_grid_fake, global_step=step)
                step += 1

            # Menyimpan model generator
            torch.save(gen.state_dict(), "train/DCGAN/train_dcgan.pth")

    # Evaluasi model pada mode evaluasi (validasi)
    gen.eval()
    disc.eval()

    val_loss_disc_total = 0.0
    val_loss_gen_total = 0.0
    num_val_batches = len(val_dataloader)

    # Loop evaluasi pada dataset validasi
    with torch.no_grad():
        for val_batch_idx, (val_real, _) in enumerate(val_dataloader):
            val_real = val_real.to(device)
            val_noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            val_fake = gen(val_noise)

            val_disc_real = disc(val_real).reshape(-1)
            val_loss_disc_real = criterion(val_disc_real, torch.ones_like(val_disc_real))

            val_disc_fake = disc(val_fake.detach()).reshape(-1)
            val_loss_disc_fake = criterion(val_disc_fake, torch.zeros_like(val_disc_fake))

            val_loss_disc = (val_loss_disc_real + val_loss_disc_fake) / 2
            val_loss_disc_total += val_loss_disc.item()

            val_output = disc(val_fake).reshape(-1)
            val_loss_gen = criterion(val_output, torch.ones_like(val_output))
            val_loss_gen_total += val_loss_gen.item()

        # Menghitung rata-rata loss pada discriminator dan generator untuk dataset validasi
        val_avg_loss_disc = val_loss_disc_total / num_val_batches
        val_avg_loss_gen = val_loss_gen_total / num_val_batches

        print(f"Validation Loss D: {val_avg_loss_disc:.4f}, Loss G: {val_avg_loss_gen:.4f}")

    # Kembali ke mode pelatihan
    gen.train()
    disc.train()
