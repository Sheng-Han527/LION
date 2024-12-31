#%% This example shows how to train FBPConvNet for full angle, noisy measurements.


#%% Imports

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

# psnr from Zak
def my_ssim(x: torch.tensor, y: torch.tensor, data_range=None):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    if data_range is None:
        data_range = x.max() - x.min()
    elif type(data_range) == torch.Tensor:
        data_range = data_range.cpu().numpy().squeeze()
    return ssim(x, y, data_range=data_range)


def my_psnr(x: torch.tensor, y: torch.tensor, data_range=None):
    x = x.detach().cpu().numpy().squeeze()
    y = y.detach().cpu().numpy().squeeze()
    if data_range is None:
        data_range = x.max() - x.min()
    elif type(data_range) == torch.Tensor:
        data_range = data_range.cpu().numpy().squeeze()
    return psnr(x, y, data_range=data_range)

def wrapper_psnr(x, y, *args,**kwargs):
    return my_psnr(y,x)
psnr_module= wrapper_psnr
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Fixed Noise t=1",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-3,
    "architecture": "FBPConvNet",
    "dataset": "LIDC",
    "epochs": 20,
    }
)


#%%
# % Chose device:
device = torch.device("cuda:3")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/local/scratch/public/sh2146/Unet1")

final_result_fname = savefolder.joinpath("FBPConvNet_fixed_noise_final_iter.pt")
checkpoint_fname = savefolder.joinpath("FBPConvNet_fixed_noise_check_*.pt")  
validation_fname = savefolder.joinpath("FBPConvNet_fixed_noise_min_val.pt")
#
# #%% Define experiment
# # experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)
# experiment = ct_experiments.clinicalCTRecon()

# #%% Dataset
# lidc_dataset = experiment.get_training_dataset()
# lidc_dataset_val = experiment.get_validation_dataset()

# #%% Define DataLoader
# # Use the same amount of training
# batch_size = 4
# lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
# lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)

# This sets the geometry
import LION.CTtools.ct_geometry as ctgeo
geo = ctgeo.Geometry.default_parameters()

# This sets the DataSet
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
data_loader_params = LIDC_IDRI.default_parameters(geo=geo, task="reconstruction")
data_loader_params.max_num_slices_per_patient = 5
train_dataset= LIDC_IDRI(mode="train",parameters=data_loader_params,geometry_parameters=geo)
validation_dataset= LIDC_IDRI(mode="validation",parameters=data_loader_params,geometry_parameters=geo)
test_dataset= LIDC_IDRI(mode="test",parameters=data_loader_params,geometry_parameters=geo)

# test_immature version
# num_training=100
# from torch.utils.data import Subset
# train_dataset=Subset(train_dataset,range(num_training))
# validation_dataset=Subset(validation_dataset,range(num_training))

# You can now set the DataLoader
batch_size=4
lidc_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
lidc_validation= DataLoader(validation_dataset, batch_size, shuffle=True)
test_data_loader= DataLoader(test_dataset, batch_size, shuffle=False)

# #%% Model
# # Default model is already from the paper.
# model = FBPConvNet(geometry_parameters=experiment.geo).to(device)
 
model = FBPConvNet(geo).to(device)

#%% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 20
train_param.learning_rate = 1e-3
train_param.betas = (0.9, 0.99)
train_param.loss = "MSELoss"
optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# learning parameter update
steps = len(lidc_dataloader)
model.train()
min_valid_loss = np.inf
total_loss = np.zeros(train_param.epochs)
start_epoch = 0

# %% Check if there is a checkpoint saved, and if so, start from there.

# If there is a file with the final results, don't run again
#if model.final_file_exists(savefolder.joinpath(final_result_fname)):
#    print("final model exists! You already reached final iter")
#    exit()



# model, optimiser, start_epoch, total_loss, _ = FBPConvNet.load_checkpoint_if_exists(
#     checkpoint_fname, model, optimiser, total_loss
# )

# total_loss = np.resize(total_loss, train_param.epochs)
# print(f"Starting iteration at epoch {start_epoch}")


# Plot sample reconstruction if final model exists
'''only for testing, this plot is shit'''
import LION.CTtools.ct_utils as ct

def Shengs_t_to_I0_converter(t):
    return 1000+10**5*np.exp(-10*t)
def plot_reconstruction( recon, target_recon):
    plt.figure()
    plt.subplot(121)
    plt.imshow(recon)
    plt.clim(0, 2.5)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(target_recon)
    plt.clim(0, 2.5)
    plt.axis('off')
    plt.savefig("fixed_modelt=1_reconstruction.png")

if model.final_file_exists(savefolder.joinpath(final_result_fname)):
    model,_ , _ =FBPConvNet.load(final_result_fname)
    model.eval()
    print("final model exists! You already reached final iter")
    for sinogram, target_reconstruction in lidc_dataloader:
        t= 1
        print(t)
        I0=Shengs_t_to_I0_converter(t)
        with torch.no_grad():
            for i in range(sinogram.shape[0]):
                sinogram[i]=ct.sinogram_add_noise(sinogram[i],I0=I0)
        sinogram.to(device)
        fdk_recon = fdk(sinogram, model.op)
        reconstruction = model(sinogram,t)
        
        loss = loss_fcn(reconstruction, target_reconstruction)
        print('model loss',loss.item())
        print('fdk loss', loss_fcn(fdk_recon, target_reconstruction).item())
        
        print(psnr_module(fdk_recon,target_reconstruction, 'mean'))
        print(print(psnr_module(reconstruction,target_reconstruction, 'mean')))
        
        fdk_recon= fdk_recon.detach().cpu().numpy()
        reconstruction = reconstruction.detach().cpu().numpy()
        target_reconstruction = target_reconstruction.detach().cpu().numpy()
        
        #calculate validation loss 
        
        plot_reconstruction(
            reconstruction[0,: , : , : ][0],target_reconstruction[0,: , : , : ][0]
            )
        plt.figure()
        plt.imshow(fdk_recon[0,: , : , : ][0])
        plt.savefig("fdk_t=1.png")
        
        exit()
# # '''plot test code ends'''



#%% train
import LION.CTtools.ct_utils as ct

def Shengs_t_to_I0_converter(t):
    return 1000+10**5*np.exp(-10*t)

for epoch in range(start_epoch, train_param.epochs):
    train_loss = 0.0
    for idx, (sinogram, target_reconstruction) in enumerate(tqdm(lidc_dataloader)):
        t= 1
        # #random from 0-1
        # t=0
        I0=Shengs_t_to_I0_converter(t)
        with torch.no_grad():
            for i in range(sinogram.shape[0]):
                sinogram[i]=ct.sinogram_add_noise(sinogram[i],I0=I0)
        sinogram.to(device)
        
        
        if idx == 0:
            fdk_recon = fdk(sinogram, model.op)
            print(psnr_module(fdk_recon,target_reconstruction, 'mean'), torch.max(sinogram), torch.max(fdk_recon))
# for epoch in range(start_epoch, train_param.epochs):
#     train_loss = 0.0
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
#     for sinogram, target_reconstruction in tqdm(lidc_dataloader):

        optimiser.zero_grad()
        reconstruction = model(sinogram,t)
        loss = loss_fcn(reconstruction, target_reconstruction)
        wandb.log({"train_loss": loss}, step=idx + epoch*len(lidc_dataloader))

        loss.backward()

        train_loss += loss.item()

        optimiser.step()
    total_loss[epoch] = train_loss
    # Validation
    valid_loss = 0.0
    model.eval()
    for sinogram, target_reconstruction in tqdm(lidc_validation):
        with torch.no_grad():
            for i in range(sinogram.shape[0]):
                sinogram[i]=ct.sinogram_add_noise(sinogram[i],I0=I0)
        sinogram.to(device)
        reconstruction = model(sinogram,t)
        loss = loss_fcn(target_reconstruction, reconstruction)
        wandb.log({"valid_loss": loss}, step=idx + epoch*len(lidc_dataloader))
        #add noise to sino
        
        valid_loss += loss.item()

    print(
        f"""Epoch {epoch+1} \t\t Noise parameter:{t} \t\t Training Loss: {train_loss / len(lidc_dataloader)} \t\t Validation Loss: {valid_loss / len(lidc_validation)}\n
         Initial PSNR {psnr_module(fdk(sinogram,model.op),target_reconstruction,'mean')} Final PSNR {psnr_module(reconstruction,target_reconstruction,'mean')}"""
    )

    if min_valid_loss > valid_loss:
        print(
            f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
        )
        min_valid_loss = valid_loss
        # Saving State Dict
        model.save(
            validation_fname,
            epoch=epoch + 1,
            training=train_param,
            loss=min_valid_loss,
        )

    # Checkpoint every 10 iters anyway
    if epoch % 10 == 0:
        model.save_checkpoint(
            pathlib.Path(str(checkpoint_fname).replace("*", f"{epoch+1:04d}")),
            epoch + 1,
            total_loss,
            optimiser,
            train_param,
        )


model.save(
    final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
)

wandb.finish()


