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

# import our model
from core_model.part_2_model import simple_D, simple_G  # call model

# for visualize net arch
from torchsummary import summary

# define something else before


if __name__ == '__main__':

    # Hyperparameters etc.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    z_dim = 64
    image_dim = 32 * 32 * 3  # 3072
    batch_size = 32
    num_epochs = 5

    # simple_G arch check
    model_SG = simple_G(z_dim, image_dim)
    # print model arch
    summary(model_SG)

    # initialize the model instance object
    disc = simple_D(image_dim).to(device)
    gen = simple_G(z_dim, image_dim).to(device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)

    # data augmentatation
    transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # dataloader
    dataset = datasets.CIFAR10(
        root="data/", transform=transforms, download=True)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True)  # Dataloader
    # discriminator_D optimizer
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)  # generator_G optimizer
    criterion = nn.BCELoss()  # ini loss function
    # unutuk visualisasi di tensor board
    writer_fake = SummaryWriter(f"logs/fake")
    writer_real = SummaryWriter(f"logs/real")
    step = 0

    # Trainer
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 3072).to(device)  # flatening 32x32x3
            batch_size = real.shape[0]

            # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # where the second option of maximizing doesn't suffer from
            # saturating gradients
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                        Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 3, 32, 32)
                    data = real.reshape(-1, 3, 32, 32)
                    img_grid_fake = torchvision.utils.make_grid(
                        fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(
                        data, normalize=True)

                    writer_fake.add_image(
                        "Hands-on Simple GAN: CIFAR Fake Images", img_grid_fake, global_step=step
                    )
                    writer_real.add_image(
                        "Hands-on Simple GAN: CIFAR Images", img_grid_real, global_step=step
                    )
                    step += 1
