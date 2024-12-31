#%% This example shows how to train FBPConvNet for full angle, noisy measurements.


#%% Imports
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib
from LION.models.post_processing.FBPConvNet_original import FBPConvNet
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from LION.classical_algorithms.fdk import fdk
from matplotlib import pyplot as plt

# import numpy as np
# import random
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import wandb
# import pathlib
# from LION.utils.parameter import LIONParameter
# import LION.experiments.ct_experiments as ct_experiments
# from LION.metrics import psnr as psnr_metric
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from LION.classical_algorithms.fdk import fdk
# import LION.CTtools.ct_utils as ct

#%%
# % Chose device:
device = torch.device("cuda:2")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/home/sh2146/LION")

final_result_fname = savefolder.joinpath("FBPConvNet_final_iter.pt")
checkpoint_fname = savefolder.joinpath("FBPConvNet_check_*.pt")  
validation_fname = savefolder.joinpath("FBPConvNet_min_val.pt")
#
#%% Define experiment
# experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)
experiment = ct_experiments.clinicalCTRecon()
#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()

#%% Define DataLoader
# Use the same amount of training
batch_size = 4
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)

#%% Model
# Default model is already from the paper.
model = FBPConvNet(geometry_parameters=experiment.geo).to(device)


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

model, optimiser, start_epoch, total_loss, _ = FBPConvNet.load_checkpoint_if_exists(
    checkpoint_fname, model, optimiser, total_loss
)

total_loss = np.resize(total_loss, train_param.epochs)
print(f"Starting iteration at epoch {start_epoch}")


model, _ , _ =FBPConvNet.load_checkpoint(final_result_fname)
# Plot sample reconstruction if final model exists
'''only for testing, this plot is shit'''
def plot_reconstruction( recon, target_recon):
    plt.figure()
    plt.subplot(121)
    plt.imshow(recon)
    plt.clim(-1, 2.5)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(target_recon)
    plt.clim(-1, 2.5)
    plt.colorbar()
    plt.savefig("fixed_modelI0=3000_reconstruction.png")


for sinogram, target_reconstruction in lidc_validation:
    with torch.no_grad():
        for i in range(sinogram.shape[0]):
            sinogram[i]=ct.sinogram_add_noise(sinogram[i],I0=3000)
    sinogram.to(device)
    fdk_recon = fdk(sinogram, model.op)
    reconstruction = model(sinogram)
    
    # print(psnr_module(fdk_recon,target_reconstruction, 'mean'))
    # print(print(psnr_module(reconstruction,target_reconstruction, 'mean')))
    
    fdk_recon= fdk_recon.detach().cpu().numpy()
    reconstruction = reconstruction.detach().cpu().numpy()
    target_reconstruction = target_reconstruction.detach().cpu().numpy()
    
    plot_reconstruction(
        reconstruction[0,: , : , : ][0],target_reconstruction[0,: , : , : ][0]
        )
    plt.figure()
    plt.imshow(fdk_recon[0,: , : , : ][0])
    plt.savefig("fdk_I0=3000.png")
    
    exit()
'''plot test code ends'''


#%% train
for epoch in range(start_epoch, train_param.epochs):
    train_loss = 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    for sinogram, target_reconstruction in tqdm(lidc_dataloader):

        optimiser.zero_grad()
        reconstruction = model(sinogram)
        loss = loss_fcn(reconstruction, target_reconstruction)

        loss.backward()

        train_loss += loss.item()

        optimiser.step()
        scheduler.step()
    total_loss[epoch] = train_loss
    # Validation
    valid_loss = 0.0
    model.eval()
    for sinogram, target_reconstruction in tqdm(lidc_validation):
        reconstruction = model(sinogram)
        loss = loss_fcn(target_reconstruction, reconstruction)
        valid_loss += loss.item()

    print(
        f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(lidc_dataloader)} \t\t Validation Loss: {valid_loss / len(lidc_validation)}"
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
            dataset=experiment.param,
        )

    # Checkpoint every 10 iters anyway
    if epoch % 10 == 0:
        model.save_checkpoint(
            pathlib.Path(str(checkpoint_fname).replace("*", f"{epoch+1:04d}")),
            epoch + 1,
            total_loss,
            optimiser,
            train_param,
            dataset=experiment.param,
        )


model.save(
    final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
    dataset=experiment.param,
)



