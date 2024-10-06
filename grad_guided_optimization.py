import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import argparse
from upload.predict_model import *
import torch.nn.functional as F
import time
import matplotlib as mpl
import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import random
from diffusion import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

from vae import *

device = "cuda:0"
ddpm = torch.load(r"model_ddpm.pth")
ddpm.eval()



ELEMENTS = ['O', 'H', 'Li', 'Be', 'B', 'C', 'N', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V',
            'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
            'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
            'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
p = predict_model()

def process_superc(superc):
    superc = torch.argmax(superc, dim=3).float()
    int_part = superc[:, 0, :]
    first_dec = superc[:, 1, :] / 10
    second_dec = superc[:, 2, :] / 100
    superc = int_part + first_dec + second_dec
    return superc

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim, n_feat=256, n_classes=4):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim, n_feat, n_classes)
        self.decoder = Decoder(in_channels, latent_dim, n_feat, n_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.encoder.latent_dim).to(device)
        samples = self.decoder(z)
        return samples
    
    def estimate_gradient_mic1(self, z, q, beta,batch_size=10):
        old_superc = self.decoder(z)
        old_Tc = p.predict(old_superc)

        u = torch.randn(q, nz).float().to(device)
        u = beta * torch.nn.functional.normalize(u, dim=-1)

        conditioned = z + u
        new_superc = self.decoder(conditioned)
        # print(new_superc.size() )
        new_superc=ddpm.sample(new_superc,device)
        new_superc = process_superc(new_superc)
        
        new_Tc = p.predict(new_superc)

        grads = torch.stack([beta * (old_Tc[0][0] - new_Tc[i]).float() * u[i] for i in range(q)], dim=0).sum(dim=0)
        return grads
    def estimate_gradient_mic(self, z, q, beta, batch_size=1):
        old_superc = self.decoder(z)
        old_Tc = p.predict(old_superc)

        grads = torch.zeros_like(z).to(device) 
        for i in range(0, q, batch_size): 
            u_batch = torch.randn(batch_size, nz).float().to(device)
            u_batch = beta * torch.nn.functional.normalize(u_batch, dim=-1)

            conditioned_batch = z + u_batch
            new_superc_batch = self.decoder(conditioned_batch)
            new_superc_batch = ddpm.sample(new_superc_batch, device)
            new_superc_batch = process_superc(new_superc_batch)
            new_Tc_batch = p.predict(new_superc_batch)

            grads_batch = torch.stack([beta * (old_Tc[0][0] - new_Tc_batch[j]).float() * u_batch[j] for j in range(batch_size)], dim=0).sum(dim=0)
            grads += grads_batch 

        return grads / q 


vae = VAE(in_channels=3, latent_dim=20, n_feat=256).to(device)
vae=torch.load("vae_model.pth")
vae.eval()
print('loading pretrained VAE model')


def get_fumal(array):
    gen_im = array.detach().cpu()
    gen_im = gen_im.view(-1, 3, 86, 10)
    array = np.array(gen_im[0])

    f = str()
    for i in range(3):
        for j in range(86):
            if float(str(array[i, j])) > 0:
                f = f + str(ELEMENTS[j]) + str(array[i, j])[:4]
    return f

def bracket_hash(formula):
    element = ""
    element_hash = {}
    for x in formula:
        if x.isupper():
            element = x
            element_count = ''
        elif x.islower():
            element += x
        else:
            element_count += x
            element_hash[element] = element_count
    return element_hash

def formula2vec(formula):
    com = bracket_hash(str(formula))
    array = {m: 0 for m in ELEMENTS}
    index = []
    for p in com:
        if p in array:
            array[p] = com[p]
        else:
            array = []
            print('=====')
    for q in array:
        index.append(array[q])
    index = np.array(index).astype(float)
    index = index.reshape(-1, 1, 86)
    index = torch.tensor(index)
    index = index.to(device).float()
    return index

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("=====> Setup model")
nz = 20

def get_fumal_gen(output_all):
    all_fumals = []
    output_all=output_all.cpu().detach().numpy()[0]
    output_all=output_all.reshape([1,3,86,10])
    for output in output_all:
        output=output.reshape([3,86,10])
        output = np.argmax(np.where(output > 0, output, 0),axis=2)

        int_output = output[0]
        dec_output = output[1]
        third_output = output[2]
        int_output_index= np.where(int_output > 0)[0]
        dec_output_index= np.where(dec_output > 0)[0]
        third_output_index= np.where(third_output > 0)[0]
        f=str()

        for index,i in enumerate( int_output_index):
            element=ELEMENTS[i]
            f=f+str(element)+str(int_output[i])
            if i in dec_output_index:
                f=f+"."+str(dec_output[i])
                if i in third_output_index:
                    f=f+str(third_output[i])
        
        for i in dec_output_index:
            if i not in int_output_index:
                element=ELEMENTS[i]
                f=f+str(element)+"0."+str(dec_output[i])
                if i in third_output_index:
                    f=f+str(third_output[i])
        
        for i in third_output_index:
            if i not in int_output_index and i not in dec_output_index:
                element=ELEMENTS[i]
                f=f+str(element)+"0.0"+str(third_output[i])

        print(f)
        all_fumals.append(f)
    all_fumals=pd.DataFrame(all_fumals)
    return all_fumals

def perform_optimization(vae, p, z_0, attempts=30, learning_rate=0.1):
    z = z_0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=learning_rate)
    traj_z = []
    traj_tc = []
    for _ in range(attempts):
        optimizer.zero_grad()
        grad = vae.module.estimate_gradient_mic(z, 30, 0.5)
        z.grad = torch.tensor(grad).to(device)
        optimizer.step()
        new_superc_ = vae.module.decoder(z)
        new_superc_ = process_superc(new_superc_)

        new_superc = new_superc_.unsqueeze(1)
        new_tc = p.predict(new_superc)
        print('new_tc:', new_tc)

        traj_z.append(z.clone().detach().cpu().numpy())
        traj_tc.append(new_tc[0].item())
    return traj_z, traj_tc, z.clone().detach().cpu().numpy()

def generate_random_walk(z_0, steps=30, step_size=0.01):
    z = z_0.clone().detach().cpu().numpy()
    random_walk = [z.copy()]
    random_walk_tc = []
    for _ in range(steps):
        direction = np.random.randn(*z.shape)
        direction /= np.linalg.norm(direction)
        z += step_size * direction
        random_walk.append(z.copy())
        new_superc = vae.module.decoder(torch.tensor(z).to(device).float())
        new_superc = process_superc(new_superc)
        new_superc = new_superc.unsqueeze(1)

        new_tc = p.predict(new_superc)
        random_walk_tc.append(new_tc[0].item())
    return random_walk, random_walk_tc

def local_grid_sampling(z_0, z_star, vae, p, grid_size=30, step_size=0.1):
    v_x = z_star - z_0
    v_x /= np.linalg.norm(v_x)
    v_y = np.random.randn(*v_x.shape)
    v_y = v_y - np.dot(v_y, v_x.T) * v_x
    v_y /= np.linalg.norm(v_y)

    grid_points = []
    for i in range(-grid_size, grid_size + 1):
        for j in range(-grid_size, grid_size + 1):
            point = z_0 + i * step_size * v_x + j * step_size * v_y
            grid_points.append(point)
    grid_points = np.array(grid_points)

    properties = []
    for point in grid_points:
        z = torch.tensor(point).to(device).float().requires_grad_(False)
        new_superc = vae.module.decoder(z)
        new_superc = process_superc(new_superc)
        new_superc = new_superc.unsqueeze(1)

        new_tc = p.predict(new_superc)
        properties.append(new_tc[0].item())
    
    print("properties:", properties)

    return grid_points, properties

def visualize_optimization_2d_contour(grid_points, properties, trajectories, random_walk, random_walk_tc):
    scaler = StandardScaler()
    grid_points_scaled = scaler.fit_transform(grid_points.reshape(grid_points.shape[0], -1))

    all_traj_scaled = [scaler.transform(np.array(traj).reshape(len(traj), -1)) for traj, _ in trajectories]
    random_walk_scaled = scaler.transform(np.array(random_walk).reshape(len(random_walk), -1))

    pca = PCA(n_components=2, random_state=seed)
    combined_data = np.vstack((grid_points_scaled, *all_traj_scaled, random_walk_scaled))
    embedded_data = pca.fit_transform(combined_data)

    grid_points_2d = embedded_data[:len(grid_points_scaled)]
    traj_2d_list = []
    start_idx = len(grid_points_scaled)
    for traj in all_traj_scaled:
        end_idx = start_idx + len(traj)
        traj_2d_list.append(embedded_data[start_idx:end_idx])
        start_idx = end_idx
    random_walk_2d = embedded_data[start_idx:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    ax1.set_aspect('equal')
    fig.subplots_adjust(wspace=0.3)

    xi = np.linspace(grid_points_2d[:, 0].min(), grid_points_2d[:, 0].max(), 200)
    yi = np.linspace(grid_points_2d[:, 1].min(), grid_points_2d[:, 1].max(), 200)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((grid_points_2d[:, 0], grid_points_2d[:, 1]), properties, (X, Y), method='linear')

    cmap = plt.cm.viridis
    contour = ax1.contourf(X, Y, Z, levels=30, cmap=cmap, vmin=min(properties), vmax=max(properties))
    ax1.contour(X, Y, Z, levels=30, colors='k', alpha=0.3, linewidths=0.5)

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#ff7f0e', '#7f7f7f', '#bcbd22', '#17becf']

    end_markers = ['s', 'o', '^', 'v', 'd', 'h']
    for idx, (traj_2d, (_, traj_tc)) in enumerate(zip(traj_2d_list, trajectories)):
        color = colors[idx]
        for i in range(len(traj_2d) - 1):
            ax1.arrow(traj_2d[i, 0], traj_2d[i, 1],
                      traj_2d[i + 1, 0] - traj_2d[i, 0],
                      traj_2d[i + 1, 1] - traj_2d[i, 1],
                      color=color, head_width=0.05, head_length=0.075, alpha=0.9)
        ax1.plot(traj_2d[-1, 0], traj_2d[-1, 1], marker=end_markers[idx], color=color, markersize=10,
                 label=f'End Optimization {idx + 1}')

    random_walk_color = '#e377c2'
    for i in range(len(random_walk_2d) - 1):
        ax1.arrow(random_walk_2d[i, 0], random_walk_2d[i, 1],
                  random_walk_2d[i + 1, 0] - random_walk_2d[i, 0],
                  random_walk_2d[i + 1, 1] - random_walk_2d[i, 1],
                  color=random_walk_color, head_width=0.05, head_length=0.075, alpha=0.9)

    ax1.plot(random_walk_2d[-1, 0], random_walk_2d[-1, 1], '*', color=random_walk_color, markersize=10,
             label='End Random Walk')

    ax1.set_xlabel('PCA Component 1', fontsize=14)
    ax1.set_ylabel('PCA Component 2', fontsize=14)
    ax1.set_title('Optimization Trajectories and Random Walk on Performance Landscape', fontsize=16)
    ax1.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.colorbar(contour, ax=ax1, label='Performance (Tc)')

    for idx, (_, traj_tc) in enumerate(trajectories):
        ax2.plot(range(len(traj_tc)), traj_tc, marker='o', color=colors[idx], label=f'Optimization {idx + 1}')
    ax2.plot(range(len(random_walk_tc)), random_walk_tc, color=random_walk_color, marker='^', label='Random Walk')
    ax2.set_xlabel('Iterations', fontsize=14)
    ax2.set_ylabel('Performance (Tc)', fontsize=14)
    ax2.set_title('Performance Comparison', fontsize=16)
    ax2.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig('2d_contour_optimization_landscape_with_8_paths_modified.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    formula = 'Y1Ba1Cu2O7'
    attempts = 30
    learning_rate = 0.1

    data = formula2vec(formula)

    int_part = data // 1
    dec_part = data % 1
    first_dec = ((dec_part) * 10) // 1
    secend_dec = (((dec_part) * 100) // 1) % 10

    data = int_part.type(torch.int64)
    x_f = first_dec.type(torch.int64)
    x_g = secend_dec.type(torch.int64)

    data = torch.nn.functional.one_hot(data, num_classes=10).type(torch.float32).unsqueeze(1)
    x_f = torch.nn.functional.one_hot(x_f, num_classes=10).type(torch.float32).unsqueeze(1)
    x_g = torch.nn.functional.one_hot(x_g, num_classes=10).type(torch.float32).unsqueeze(1)

    data = torch.cat([data, x_f, x_g], dim=1).to(device)
    data = data.squeeze(2)

    mean, logstd = vae.module.encoder(data)
    z_=vae.module.reparameterize(mean, logstd)

    initial_z_0 = z_.detach().clone().cpu().numpy()
    print(mean)

    trajectories = []
    z_0 = torch.tensor(initial_z_0).to(device).float().requires_grad_(True)

    for _ in range(6):
        traj_z, traj_tc, z_star, = perform_optimization(vae, p, z_0, attempts=attempts, learning_rate=learning_rate)
        trajectories.append((traj_z, traj_tc))

    z_0 = torch.tensor(initial_z_0).to(device).float().requires_grad_(True)
    random_walk, random_walk_tc = generate_random_walk(z_0, steps=attempts, step_size=learning_rate)

    grid_points, properties = local_grid_sampling(initial_z_0, trajectories[0][0][-1], vae, p, grid_size=attempts * 2, step_size=learning_rate)

    visualize_optimization_2d_contour(grid_points, properties, trajectories, random_walk, random_walk_tc)

if __name__ == "__main__":
    main()
