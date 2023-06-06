import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from core_model.part_2_model import simple_D, simple_G

if __name__ == '__main__':
    # Hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    z_dim = 64
    image_dim = 32 * 32 * 3
    batch_size = 32
    num_epochs = 5

    # Define data augmentation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root='data/MyDataset', transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    model_D = simple_D(image_dim).to(device)
    model_G = simple_G(z_dim, image_dim).to(device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)

    # Print model architectures
    summary(model_D, (image_dim,))
    summary(model_G, (z_dim,))

    # Optimizers and loss function
    opt_D = optim.Adam(model_D.parameters(), lr=lr)
    opt_G = optim.Adam(model_G.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Tensorboard writer
    writer_fake = SummaryWriter(f"logs/fake")
    writer_real = SummaryWriter(f"logs/real")
    step = 0

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, image_dim).to(device)
            batch_size = real.shape[0]

            # Train Discriminator
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = model_G(noise)
            output_real = model_D(real).view(-1)
            loss_D_real = criterion(output_real, torch.ones_like(output_real))
            output_fake = model_D(fake.detach()).view(-1)
            loss_D_fake = criterion(output_fake, torch.zeros_like(output_fake))
            loss_D = (loss_D_real + loss_D_fake) / 2

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train Generator
            output_fake = model_D(fake).view(-1)
            loss_G = criterion(output_fake, torch.ones_like(output_fake))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(loader)} "
                    f"Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}"
                )

                with torch.no_grad():
                    fake_images = model_G(fixed_noise).reshape(-1, 3, 32, 32)
                    real_images = real.reshape(-1, 3, 32, 32)
                    img_grid_fake = torchvision.utils.make_grid(fake_images, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(real_images, normalize=True)

                    writer_fake.add_image(
                        "GAN Fake Images", img_grid_fake, global_step=step
                    )
                    writer_real.add_image(
                        "GAN Real Images", img_grid_real, global_step=step
                    )
                    step += 1