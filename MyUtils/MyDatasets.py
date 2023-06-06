# Kelompok 3
# 412020001 - Nico Sanjaya
# 412020008 - Cristha Patrisya Pentury
# 412020009 - Yohanes Stefanus

import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class UkridaDataset_v1(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs  # Direktori root yang berisi data gambar
        self.transform = transform  # Transformasi yang akan diterapkan pada gambar
        self.list_files = []  # Daftar file gambar yang akan dimuat

        # Meloopi direktori root untuk mengumpulkan daftar file gambar
        for root_dir in self.root_dirs:
            files = os.listdir(root_dir)
            files = [f for f in files if f.endswith(".png")]
            self.list_files.extend([(root_dir, f) for f in files])

    def __len__(self):
        return len(self.list_files)  # Mengembalikan jumlah total data dalam dataset

    def __getitem__(self, index):
        root_dir, img_file = self.list_files[index]  # Mengambil direktori dan nama file gambar
        img_path = os.path.join(root_dir, img_file)  # Menggabungkan direktori dan nama file untuk mendapatkan path gambar
        image = Image.open(img_path).convert("RGB")  # Membuka gambar dan mengonversi ke mode RGB

        if self.transform is not None:
            image = self.transform(image)  # Melakukan transformasi pada gambar jika transformasi diberikan

        label = self.create_label(root_dir)  # Membuat label berdasarkan direktori root

        return image, label  # Mengembalikan gambar dan label

    def create_label(self, root_dir):
        if "Bakso" in root_dir:
            return 1  # Jika direktori root mengandung "Bakso", maka labelnya adalah 1
        elif "Batagor" in root_dir:
            return 2  # Jika direktori root mengandung "Batagor", maka labelnya adalah 2
        elif "Mie Ayam" in root_dir:
            return 3  # Jika direktori root mengandung "Mie Ayam", maka labelnya adalah 3
        else:
            return 4  # Jika tidak ada kecocokan, maka labelnya adalah 4

if __name__ == "__main__":
    root_dirs = ["../data/MyDataset/Bakso", "../data/MyDataset/Batagor", "../data/MyDataset/Mie Ayam"]
    dataset = UkridaDataset_v1(root_dirs)  # Membuat objek dataset
    labels = []
    for root_dir, _ in dataset.list_files:
        label = dataset.create_label(root_dir)  # Membuat label untuk setiap direktori root dalam daftar file dataset
        labels.append(label)  # Menyimpan label dalam daftar labels
    print(labels)  # Mencetak daftar label

