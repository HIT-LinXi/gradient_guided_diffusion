import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define the periodic table elements
ELEMENTS = ['O', 'H', 'Li', 'Be', 'B', 'C', 'N', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V',
            'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
            'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
            'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']

class ChemicalFormulaDataset():
    def __init__(self, csv_path="element.csv"):
        self.data = np.array(pd.read_csv(csv_path))[:, 1:]
        print("Dataset shape:", self.data.shape)

    def __getitem__(self, item):
        return self.data[item][:-1], self.classify_data(self.data[item][-1])

    def classify_data(self, data):
        if data >= 77:
            return 3
        elif data > 40:
            return 2
        else:
            return 1

    def __len__(self):
        return len(self.data)


def decode_formula(output_all):
    """Decodes the one-hot encoded output to a chemical formula string."""
    formulas = []
    output_all = output_all.cpu().detach().numpy()[0].reshape(3, 86, 10)
    for output in output_all:
        output = np.argmax(np.where(output > 0, output, 0), axis=1)
        formula = "".join([f"{ELEMENTS[i]}{val}" for i, val in enumerate(output) if val > 0])
        formulas.append(formula)
    return pd.DataFrame(formulas)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                   nn.BatchNorm2d(out_channels), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                   nn.BatchNorm2d(out_channels), nn.GELU())

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        out = (x + x2) / 1.414 if self.is_res and self.same_channels else x1 + x2 if self.is_res else x2
        return out


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2))

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):  # Merged UnetUp1 and UnetUp2 - they were almost identical
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, n_feat=256, n_classes=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(2), nn.GELU())
        self.fc_mu = nn.Linear(2 * n_feat * 10 * 1, latent_dim)
        self.fc_logvar = nn.Linear(2 * n_feat * 10 * 1, latent_dim)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.to_vec(x).view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dim, n_feat=256, n_classes=4):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 2 * n_feat * 10 * 1)
        self.up0 = nn.Sequential(nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, [3,2], 2, 0),
                                 nn.GroupNorm(8, 2 * n_feat), nn.ReLU())
        self.up1 = UnetUp(4 * n_feat, n_feat, kernel_size=[3,2], stride=2)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), nn.GroupNorm(8, n_feat),
                                 nn.Sigmoid(), nn.Conv2d(n_feat, out_channels, 3, 1, 1))

    def forward(self, z):
        z = self.fc(z).view(-1, self.fc.out_features // (10*1) , 10, 1) # use out_features for dynamic shape
        z = self.up0(z)
        z = self.up1(z, z)  # No skip connection, using same tensor twice
        z = self.up2(z, z)  # No skip connection
        return self.out(torch.cat((z, z), 1))  # No skip connection


class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim, n_feat=256, n_classes=4):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, n_feat, n_classes)
        self.decoder = Decoder(in_channels, latent_dim, n_feat, n_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.encoder.latent_dim).to(device)
        return self.decoder(z)


def train_vae(n_epoch=100000, batch_size=512, device="cuda:0", n_feat=256, lrate=1e-4, save_model=True, load_model=0):
    vae = VAE(in_channels=3, latent_dim=128, n_feat=n_feat).to(device)
    vae = torch.nn.DataParallel(vae, device_ids=range(torch.cuda.device_count()))

    if load_model:
        vae = torch.load(r"model/vae_model_150.pth")
        print('Loaded model')

    dataset = ChemicalFormulaDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    optim = torch.optim.Adam(vae.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'Epoch {ep}')
        vae.train()
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)  # linear learning rate decay

        pbar = tqdm(dataloader)
        loss_ema = None

        for x, c in pbar:
            x = torch.round(x, decimals=2)
            int_part, dec_part = x.div(1, rounding_mode='floor'), x % 1
            first_dec = (dec_part * 10).div(1, rounding_mode='floor')
            second_dec = (dec_part * 100).div(1, rounding_mode='floor') % 10

            x = F.one_hot(int_part.long(), num_classes=10).float().unsqueeze(1).to(device)
            x_f = F.one_hot(first_dec.long(), num_classes=10).float().unsqueeze(1).to(device)
            x_g = F.one_hot(second_dec.long(), num_classes=10).float().unsqueeze(1).to(device)

            input_tensor = torch.cat([x, x_f, x_g], dim=1)

            optim.zero_grad()
            x_recon, mu, logvar = vae(input_tensor)

            recon_loss = F.binary_cross_entropy_with_logits(x_recon, input_tensor, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            loss.backward()

            loss_ema = loss.item() if loss_ema is None else 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        if ep % 50 == 0:
            print('Evaluating')
            vae.eval()
            with torch.no_grad():
                samples = vae.module.sample(1, device)
                all_f = decode_formula(samples.detach().cpu())

        if save_model and ep % 50 == 0:
            torch.save(vae, f"model/vae_model_{ep}.pth")
            print('Saved model at model.pth')


if __name__ == "__main__":
    train_vae()

