# Kelompok 3
# 412020001 - Nico Sanjaya
# 412020008 - Cristha Patrisya Pentury
# 412020009 - Yohanes Stefanus

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from MyUtils.MyDatasets import UkridaDataset_v1
from core_model.part_1_model import YourOwnCNN
from PIL import Image

torch.manual_seed(0)

num_epochs = 50
batch_size = 8
learning_rate = 0.001
validation_split = 0.2

# Transformasi yang diterapkan pada gambar
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Mengubah ukuran gambar menjadi 32x32 piksel
    transforms.ToTensor(),  # Mengonversi gambar menjadi tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisasi gambar dengan mean dan standard deviation
])

# Direktori root untuk dataset pelatihan dan validasi
train_root_dirs = ["data/ukrida_dataset/train/Bakso", "data/ukrida_dataset/train/Batagor", "data/ukrida_dataset/train/Mie Ayam"]
val_root_dirs = ["data/ukrida_dataset/val/Bakso", "data/ukrida_dataset/val/Batagor", "data/ukrida_dataset/val/Mie Ayam"]

# Membuat dataset pelatihan
train_dataset = UkridaDataset_v1(train_root_dirs, transform=transform)

# Memeriksa apakah dataset pelatihan kosong
if len(train_dataset) == 0:
    print("No data found in the training dataset. Check if the dataset is empty or the path is correct.")
    exit()

# Menghitung jumlah data untuk dataset validasi
val_size = int(validation_split * len(train_dataset))
train_size = len(train_dataset) - val_size

# Membagi dataset pelatihan menjadi dataset pelatihan dan validasi berdasarkan persentase yang ditentukan
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Membuat data loader untuk pelatihan dan validasi
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Membuat model CNN
model = YourOwnCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Menggunakan fungsi loss CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
# Menggunakan optimizer Adam untuk mengoptimasi parameter model
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loop untuk setiap epoch pelatihan
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Loop untuk setiap batch dalam data loader pelatihan
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Menghitung output dari model
        outputs = model(images)
        # Menghitung loss berdasarkan output dan label
        loss = criterion(outputs, labels)
        # Menghitung gradien dan melakukan backpropagation
        loss.backward()
        # Mengupdate parameter model menggunakan optimizer
        optimizer.step()
        
        # Menghitung loss rata-rata dalam satu epoch
        running_loss += loss.item()
        # Menghitung jumlah prediksi yang benar dan total data
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_dataloader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Menyimpan parameter model setelah pelatihan
torch.save(model.state_dict(), "train/CNN/train_CNN.pth")
print("Training finished. Model saved.")
