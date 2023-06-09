import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from core_model.part_2_model import simple_G, simple_D, DCGAN_G, DCGAN_D, initialize_weights

from torchsummary import summary

if __name__ == '__main__':
    #

    # Hyperparameters etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # modify to yaml configuration mode
    LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
    BATCH_SIZE = 32
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3
    NOISE_DIM = 100
    NUM_EPOCHS = 5
    FEATURES_DISC = 64
    FEATURES_GEN = 64

    model_ = DCGAN_G(channels_img=CHANNELS_IMG,
                     features_g=FEATURES_GEN, channels_noise=NOISE_DIM)
    # visualize net arch parameters
    summary(model_,)

    transforms = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [
                    0.5 for _ in range(CHANNELS_IMG)]
            ),
        ]
    )

    # If you train on MNIST, remember to set channels_img to 1 and CIFAR-10 cahnnels_img to 3
    dataset = datasets.CIFAR10(
        root="data/", train=True, transform=transforms, download=True
    )

    # comment mnist above and uncomment below if train on CelebA
    # dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    gen = DCGAN_G(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = DCGAN_D(CHANNELS_IMG, FEATURES_DISC).to(device)

    # weight inisial
    initialize_weights(gen)
    initialize_weights(disc)

    # optimizer for training
    opt_gen = optim.Adam(
        gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(
        disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()  # loss function
    fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
    # log file for TensorBoard
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0

    gen.train()
    disc.train()

    for epoch in range(NUM_EPOCHS):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            fake = gen(noise)

            # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                    Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True)

                    writer_real.add_image(
                        "DCGAN Real", img_grid_real, global_step=step)
                    writer_fake.add_image(
                        "DCGAN Fake", img_grid_fake, global_step=step)

                step += 1
