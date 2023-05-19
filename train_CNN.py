import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from MyUtils.MyDatasets import UkridaDataset_v1
from core_model.part_1_model import YourOwnCNN
from PIL import Image

# Mendefinisikan parameter-parameter pelatihan
num_epochs = 8
batch_size = 8
learning_rate = 0.001
validation_split = 0.2 

# Memeriksa apakah CUDA (GPU) tersedia dan mengatur perangkat yang akan digunakan
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mentransformasi gambar dengan resize, konversi ke tensor, dan normalisasi
transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Mendefinisikan daftar direktori root dataset
root_dirs = ["data/MyDataset/Bakso", "data/MyDataset/Batagor", "data/MyDataset/Mie Ayam"]

# Membuat objek dataset berdasarkan direktori root dan transformasi yang telah didefinisikan
dataset = UkridaDataset_v1(root_dirs, transform=transform) 

# Memeriksa apakah dataset kosong dan keluar dari program
if len(dataset) == 0:
    print("No data found in the dataset. Check if the dataset is empty or the path is correct.")
    exit()

# Menghitung ukuran dataset validasi berdasarkan persentase pembagian
val_size = int(validation_split * len(dataset))
train_size = len(dataset) - val_size

# Membagi dataset menjadi subset pelatihan dan validasi secara acak
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Membuat data loader untuk subset pelatihan dan validasi
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Membuat objek model CNN yang telah didefinisikan dan memindahkan model ke perangkat yang ditentukan
model = YourOwnCNN().to(device)

# Menentukan fungsi loss (CrossEntropyLoss) dan algoritma optimizer (Adam)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Menghitung jumlah langkah pelatihan berdasarkan data loader
total_step = len(train_dataloader)

# Memeriksa apakah data loader pelatihan kosong dan keluar dari program
if total_step == 0:
    print("No data found in the train dataloader. Check if the dataset is empty or the path is correct.")
    exit()

# Melakukan pelatihan model selama jumlah epoch yang telah ditentukan
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Iterasi melalui setiap batch data dalam data loader pelatihan
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Melakukan forward pass (menghasilkan output model)
        outputs = model(images)

        # Menghitung nilai loss berdasarkan output dan label
        loss = criterion(outputs, labels)

        # Menghapus gradien sebelum melakukan backpropagation
        optimizer.zero_grad()

        # Melakukan backpropagation (menghitung gradien)
        loss.backward()

        # Melakukan update parameter berdasarkan gradien menggunakan optimizer
        optimizer.step()

        running_loss += loss.item()

        # Menghitung jumlah prediksi yang benar dan total label
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Menampilkan informasi loss setiap 100 langkah
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")

    # Menghitung rata-rata loss dan akurasi epoch
    epoch_loss = running_loss / len(train_dataloader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    val_correct = 0
    val_total = 0

    # Evaluasi model pada data validasi tanpa perhitungan gradien
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            val_outputs = model(images)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (val_predicted == labels).sum().item()

# Menghitung akurasi akhir pada data pelatihan
final_accuracy = 100 * correct / total
print(f"Training finished. Final Accuracy: {final_accuracy:.2f}%")
