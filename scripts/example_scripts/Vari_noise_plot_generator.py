# Standard imports
import os
import matplotlib.pyplot as plt
import pathlib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from matplotlib import pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import pathlib
from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from LION.metrics import psnr as psnr_metric
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from LION.classical_algorithms.fdk import fdk

# Torch imports
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from tqdm import tqdm
from LION.utils.math import power_method

# Lion imports
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_benchmarking_experiments as ct_benchmarking
from LION.models.learned_regularizer.ACR import ACR
from LION.models.learned_regularizer.AR import AR
# from LION.models.learned_regularizer.TDV import TDV
# from LION.models.learned_regularizer.TDV_files.model import L2DenoiseDataterm

from ts_algorithms import sirt, tv_min, nag_ls

from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.CTtools.ct_utils import make_operator
import LION.CTtools.ct_geometry as ctgeo

geo = ctgeo.Geometry.default_parameters()

# Just a temporary SSIM that takes troch tensors (will be added to LION at some point)
def my_ssim(x: torch.tensor, y: torch.tensor, data_range=None):
    if type(x) == torch.Tensor:
        x = x.detach().cpu().numpy().squeeze()
        y = y.detach().cpu().numpy().squeeze()
    if data_range is None:
        data_range = x.max() - x.min()
    elif type(data_range) == torch.Tensor:
        data_range = data_range.cpu().numpy().squeeze()
    return ssim(x, y, data_range=data_range)


def my_psnr(x: torch.tensor, y: torch.tensor, data_range=None):
    if type(x) == torch.Tensor:
        x = x.detach().cpu().numpy().squeeze()
        y = y.detach().cpu().numpy().squeeze()
    if data_range is None:
        data_range = x.max() - x.min()
    elif type(data_range) == torch.Tensor:
        data_range = data_range.cpu().numpy().squeeze()
    return psnr(x, y, data_range=data_range)
def wrapper_psnr(x, y, *args,**kwargs):
    return my_psnr(y,x)
def wrapper_ssim(x, y, *args,**kwargs):
    return my_ssim(y,x)
psnr_module= wrapper_psnr
ssim_module= wrapper_ssim

device = torch.device("cuda:2")
torch.cuda.set_device(device)

data_loader_params = LIDC_IDRI.default_parameters(geo=geo, task="reconstruction")
data_loader_params.max_num_slices_per_patient = 5
test_dataset= LIDC_IDRI(mode="test",parameters=data_loader_params,geometry_parameters=geo)
batch_size=1
validation_dataset= LIDC_IDRI(mode="validation",parameters=data_loader_params,geometry_parameters=geo)
lidc_validation= DataLoader(validation_dataset, batch_size, shuffle=True)



# from torch.utils.data import Subset
# test_dataset= Subset(test_dataset,range(16))

test_data_loader= DataLoader(test_dataset, batch_size, shuffle=False)


import LION.CTtools.ct_utils as ct

def Shengs_t_to_I0_converter(t):
    return 1000+10**5*np.exp(-10*t)

loss_fcn = torch.nn.MSELoss()
model = FBPConvNet(geo).to(device)

noise_params = 0.5
vari_min_val_path=pathlib.Path('/local/scratch/public/sh2146/UnetVari/FBPConvNet_vari_noise_check_0011.pt')
fixed_min_val_path=pathlib.Path(f'/local/scratch/public/sh2146/Unet{noise_params}/FBPConvNet_fixed_noise_min_val.pt')

def plot_for_paper(ground_truth, noisy, vari_noise_recon, fixed_noise_recon):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    images = [ground_truth, noisy, vari_noise_recon, fixed_noise_recon]
    titles = ['Ground Truth', 'Noisy', 'Vari Noise Recon', 'Fixed Noise Recon']
    
    for i, (image, title) in enumerate(zip(images, titles)):
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis("off")
        if i != 0:  # Skip PSNR and SSIM for the ground truth
            psnr_value = psnr_module(image, ground_truth, 'mean')
            ssim_value = ssim_module(image, ground_truth, 'mean')
            axes[i].text(0.01, 0.99, f'PSNR: {psnr_value:.2f}\nSSIM: {ssim_value:.2f}', 
                         transform=axes[i].transAxes, fontsize=10, verticalalignment='top', 
                         bbox=dict(facecolor='white', alpha=0.5))
    
    plt.savefig(f"/home/sh2146/LION/vari_fixed_recon001_noise={noise_params}.png")
    
model, _ , _ = FBPConvNet.load(vari_min_val_path)
model.eval()

fixed_model, _ , _ = FBPConvNet.load(fixed_min_val_path)
fixed_model.eval()


'''plot test code starts'''
for sinogram, target_reconstruction in test_data_loader:
    t = noise_params
    print(t)
    I0 = Shengs_t_to_I0_converter(t)
    with torch.no_grad():
        for i in range(sinogram.shape[0]):
            sinogram[i] = ct.sinogram_add_noise(sinogram[i], I0=I0)
    sinogram.to(device)
    
    reconstruction = model(sinogram, t)
    fixed_reconstruction = fixed_model(sinogram, t)
    fdk_recon = fdk(sinogram, model.op)
    loss = loss_fcn(reconstruction, target_reconstruction)
    print('model loss', loss.item())
    print('fdk loss', loss_fcn(fdk_recon, target_reconstruction).item())
    
    print(ssim_module(fdk_recon[0][0], target_reconstruction[0][0], 'mean'))
    print(ssim_module(reconstruction[0][0], target_reconstruction[0][0], 'mean'))
    
    print(psnr_module(fdk_recon, target_reconstruction, 'mean'))
    print(psnr_module(reconstruction, target_reconstruction, 'mean'))
    
    fdk_recon = fdk_recon.detach().cpu().numpy()
    reconstruction = reconstruction.detach().cpu().numpy()
    fixed_reconstruction = fixed_reconstruction.detach().cpu().numpy()
    target_reconstruction = target_reconstruction.detach().cpu().numpy()
    print(reconstruction.shape)
        
    plot_for_paper(target_reconstruction[0][0], fdk_recon[0][0], reconstruction[0][0], fixed_reconstruction[0][0])    
    exit()
'''plot test code ends'''

file_path_psnr = '/home/sh2146/LION/plots_for_paper/vari_noise_psnr_mean.npy'
file_path_ssim = '/home/sh2146/LION/plots_for_paper/vari_noise_ssim_mean.npy'



if os.path.exists(file_path_psnr):
    print(f"{file_path_psnr} already exists. Plot with existing data.")
    T=np.linspace(0,1,101)
    vari_noise_psnr_mean = np.load(file_path_psnr, allow_pickle=True)
    vari_noise_ssim_mean = np.load(file_path_ssim, allow_pickle=True)
else:
    '''generating vari_noise_recon PSNR and SSIM data for all test data, this may take a while'''
    T=np.linspace(0,1,101)
    print(len(test_data_loader))
    vari_noise_psnr_values = []
    vari_noise_ssim_values = []
    vari_noise_psnr_mean=[]
    vari_noise_ssim_mean=[]
    for t in tqdm(T):
        ssim_values = []
        psnr_values = []
        dummy=1
        for sinogram, target_reconstruction in test_data_loader:
            I0 = Shengs_t_to_I0_converter(t)
            with torch.no_grad():
                for i in range(sinogram.shape[0]):
                    sinogram[i] = ct.sinogram_add_noise(sinogram[i], I0=I0)
            sinogram.to(device)
            
            with torch.no_grad():
                reconstruction = model(sinogram, t)
            # reconstruction = reconstruction.detach().cpu().numpy().squeeze()
            # target_reconstruction = target_reconstruction.detach().cpu().numpy().squeeze()
            # print(f"Reconstruction device: {reconstruction.device}")
            # print(f"Sinogram device: {sinogram.device}")
            # print(f"Target reconstruction device: {target_reconstruction.device}")
            
            psnr_values.append(psnr_module(reconstruction, target_reconstruction, 'mean'))
            ssim_values.append(ssim_module(reconstruction, target_reconstruction, 'mean'))
            # if dummy==1:
            #     print(psnr_values[-1],ssim_values[-1])
            #     dummy=0
        vari_noise_psnr_mean.append(np.mean(psnr_values))
        vari_noise_ssim_mean.append(np.mean(ssim_values))
        vari_noise_psnr_values.append(psnr_values)
        vari_noise_ssim_values.append(ssim_values)
        


    # Save the results to a file
    np.save('/home/sh2146/LION/plots_for_paper/vari_noise_psnr_values.npy', vari_noise_psnr_values)
    np.save('/home/sh2146/LION/plots_for_paper/vari_noise_ssim_values.npy', vari_noise_ssim_values)
    np.save('/home/sh2146/LION/plots_for_paper/vari_noise_psnr_mean.npy', vari_noise_psnr_mean)
    np.save('/home/sh2146/LION/plots_for_paper/vari_noise_ssim_mean.npy', vari_noise_ssim_mean)



'''plotting vari_noise_recon PSNR and SSIM data'''

T=np.linspace(0,1,101)
vari_noise_psnr_mean = np.load(file_path_psnr, allow_pickle=True)
vari_noise_ssim_mean = np.load(file_path_ssim, allow_pickle=True)

# plt.figure(figsize=(10, 5))
# plt.plot(T, vari_noise_psnr_mean, label='PSNR')
# plt.xlabel('T')
# plt.ylabel('PSNR')
# plt.title('PSNR vs T')
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig(f"/home/sh2146/LION/plots_for_paper/vari_noise_recon_psnr.png")

# # Plotting SSIM values
# plt.figure(figsize=(10, 5))
# plt.plot(T, vari_noise_ssim_mean, label='SSIM')
# plt.xlabel('T')
# plt.ylabel('SSIM')
# plt.title('SSIM vs T')
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig(f"/home/sh2146/LION/plots_for_paper/vari_noise_recon_ssim.png")

# exit()

fixed_data_path_psnr = '/home/sh2146/LION/plots_for_paper/fixed_noise_psnr_values.npy'
fixed_data_path_ssim = '/home/sh2146/LION/plots_for_paper/fixed_noise_ssim_values.npy'

if os.path.exists(fixed_data_path_psnr):
    print(f"{fixed_data_path_psnr} already exists. Plot with existing data.")
    T_fixed=[0, 0.25, 0.5, 0.75, 1]
    fixed_noise_psnr_values = np.load(fixed_data_path_psnr, allow_pickle=True)
    fixed_noise_ssim_values = np.load(fixed_data_path_ssim, allow_pickle=True)
else:
    '''generating fixed_noise_recon PSNR and SSIM data'''
    T_fixed=[0, 0.25, 0.5, 0.75, 1]
    fixed_noise_psnr_values = []
    fixed_noise_ssim_values = []
    for t in tqdm(T_fixed):
        fixed_min_val_path=pathlib.Path(f'/local/scratch/public/sh2146/Unet{t}/FBPConvNet_fixed_noise_min_val.pt')
        fixed_model, _ , _ = FBPConvNet.load(fixed_min_val_path)
        fixed_model.eval()
        ssim_values = []
        psnr_values = []
        dummy=1
        for sinogram, target_reconstruction in test_data_loader:
            I0 = Shengs_t_to_I0_converter(t)
            with torch.no_grad():
                for i in range(sinogram.shape[0]):
                    sinogram[i] = ct.sinogram_add_noise(sinogram[i], I0=I0)
            sinogram.to(device)
            
            with torch.no_grad():
                reconstruction = fixed_model(sinogram, t)
            # reconstruction = reconstruction.detach().cpu().numpy().squeeze()
            # target_reconstruction = target_reconstruction.detach().cpu().numpy().squeeze()
            # print(f"Reconstruction device: {reconstruction.device}")
            # print(f"Sinogram device: {sinogram.device}")
            # print(f"Target reconstruction device: {target_reconstruction.device}")
            
            psnr_values.append(psnr_module(reconstruction, target_reconstruction, 'mean'))
            ssim_values.append(ssim_module(reconstruction, target_reconstruction, 'mean'))
            if dummy==1:
                print(psnr_values[-1],ssim_values[-1])
                dummy=0

        fixed_noise_psnr_values.append(psnr_values)
        fixed_noise_ssim_values.append(ssim_values)

    print(np.array(fixed_noise_psnr_values).shape)
    # Save the results to a file
    np.save('/home/sh2146/LION/plots_for_paper/fixed_noise_psnr_values.npy', fixed_noise_psnr_values)
    np.save('/home/sh2146/LION/plots_for_paper/fixed_noise_ssim_values.npy', fixed_noise_ssim_values)        

for er_upper , er_lower in [(60, 40), (75, 25)]: 

    # Calculate mean and 50 percentile error bars
    psnr_means = [np.mean(values) for values in fixed_noise_psnr_values]
    psnr_errors = [np.percentile(values, er_upper) - np.percentile(values, er_lower) for values in fixed_noise_psnr_values]
    ssim_means = [np.mean(values) for values in fixed_noise_ssim_values]
    ssim_errors = [np.percentile(values, er_upper) - np.percentile(values, er_lower) for values in fixed_noise_ssim_values]

    # Plotting PSNR values
    plt.figure(figsize=(10, 5))
    plt.errorbar(T_fixed, psnr_means, yerr=psnr_errors, fmt='o', capsize=5, label='Fixed Noise Model')
    plt.xlabel('Noise Level')
    plt.ylabel('PSNR')
    plt.plot(T, vari_noise_psnr_mean, label='Vari Noise Model')
    plt.title(f'Mean PSNR with {er_lower}/{er_upper} Percentile Error Bars')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/sh2146/LION/plots_for_paper/vari_psnr_with_fixed_psnr_{er_lower}{er_upper}.png")

    # Plotting SSIM values
    plt.figure(figsize=(10, 5))
    plt.errorbar(T_fixed, ssim_means, yerr=ssim_errors, fmt='o', capsize=5, label='Fixed Noise Model')
    plt.xlabel('Noise Level')
    plt.ylabel('SSIM')
    plt.plot(T, vari_noise_ssim_mean, label='Vari Noise Model')
    plt.title(f'Mean SSIM with {er_lower}/{er_upper} Percentile Error Bars')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/sh2146/LION/plots_for_paper/vari_ssim_with_fixed_ssim_{er_lower}{er_upper}.png")

exit()
