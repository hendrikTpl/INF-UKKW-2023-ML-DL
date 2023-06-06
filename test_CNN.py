# Kelompok 3
# 412020001 - Nico Sanjaya
# 412020008 - Cristha Patrisya Pentury
# 412020009 - Yohanes Stefanus

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from MyUtils.MyDatasets import UkridaDataset_v1
from core_model.part_1_model import YourOwnCNN
from torch.utils.data import DataLoader

torch.manual_seed(0)

# Mendefinisikan transformasi yang diterapkan pada gambar
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Mengubah ukuran gambar menjadi 32x32 piksel
    transforms.ToTensor(),  # Mengonversi gambar menjadi tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisasi gambar dengan mean dan standard deviation
])

# Mendefinisikan direktori root untuk dataset validasi
val_root_dirs = ["data/ukrida_dataset/test/Bakso", "data/ukrida_dataset/test/Batagor", "data/ukrida_dataset/test/Mie Ayam"]

# Membuat dataset validasi
val_dataset = UkridaDataset_v1(val_root_dirs, transform=transform)

# Memeriksa apakah dataset validasi kosong
if len(val_dataset) == 0:
    print("No data found in the validation dataset. Check if the dataset is empty or the path is correct.")
    exit()

# Membuat data loader untuk validasi
batch_size = 8
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Membuat model CNN dan memindahkannya ke device yang sesuai (GPU jika tersedia, jika tidak CPU)
model = YourOwnCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("train/CNN/train_CNN.pth", map_location=device))
model.to(device)
model.eval()

# Loop pengujian
correct = 0
total = 0

# Tidak perlu perhitungan gradien dalam mode evaluasi (eval)
with torch.no_grad():
    for images, labels in val_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Menghitung akurasi
accuracy = 100 * correct / total
print(f"Accuracy on validation dataset: {accuracy:.2f}%")