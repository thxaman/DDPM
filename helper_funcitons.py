from asyncio import constants
import time
import math
from typing import Tuple, Optional
from scipy import constants
import torch 
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import device, nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import os
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import copy
from PIL import Image


# to make a ema version better and smoother weights
class EMA:
    def __init__(self, model, decay=0.995):
        """
        decay: 0.995 is a good default for ~100k-150k steps. 
               (If training for 500k+ steps, you'd use 0.9999)
        """
        self.decay = decay
       
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval() 
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        """
        Call this every batch after optimizer.step()
        """
        
        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)
            
        for ema_buffer, model_buffer in zip(self.ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(model_buffer)

    def copy_to(self, model):
        """Copies the EMA weights into the target model (useful for inference)"""
        model.load_state_dict(self.ema_model.state_dict())

def get_new_path_model(dataset_name="dataset"):
    base = "models"
    file_path = os.path.join(base, dataset_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        

    existing_files = os.listdir(file_path)
    run_ID = max(int(i.split(".")[0].split("_")[-1]) for i in existing_files)+1 if existing_files else 0
    run_name = os.path.join(file_path,f"run_{run_ID}")
    os.makedirs(run_name)
    os.makedirs(os.path.join(run_name,"checkpoints"))
    os.makedirs(os.path.join(run_name,"ema_checkpoints"))

    return run_name


def get_ddpm_constants(start_beta=0.0001, end_beta=0.02,timestep = 1000,device="cpu"):
    beta = torch.linspace(start_beta,end_beta,timestep,device=device) # noise schedule
    alpha = 1 - beta
    sqrt_alpha = torch.sqrt(alpha)
    alpha_bar = torch.cumprod(alpha,dim=0) # mu
    
    alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)


    sqrt_alpha_bar = torch.sqrt(alpha_bar) # sqrt mu
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar) # sigma var

    posterior_variance = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)

    return {"beta":beta, "alpha": alpha, 
            "alpha_bar": alpha_bar, 
            "sqrt_alpha": sqrt_alpha,
            "sqrt_alpha_bar": sqrt_alpha_bar ,
            "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
            "posterior_variance": posterior_variance,
            "alpha_bar_prev": alpha_bar_prev}

def q_sample(x0,t,constants,device=None):
    device = torch.device(device if device else "cpu")
    epsilon = torch.randn_like(x0).to(device)
    srqt_alpha_bar_t = constants["sqrt_alpha_bar"][t].reshape(-1,1,1,1).to(device)
    srqt_one_minus_alpha_bar_t = constants["sqrt_one_minus_alpha_bar"][t].reshape(-1,1,1,1).to(device)

    xt = srqt_alpha_bar_t*x0 + srqt_one_minus_alpha_bar_t*epsilon #reparameterised fromula (4)
    xt = xt.to(device)

    return xt,epsilon

#data grabbing
def get_data(
    dataset_name="mnist",
    batch_size=256,
    image_size=64,
    data_root="./data",
    num_workers=2,
    to_rgb=True,
):
    
    dataset_name = dataset_name.lower()


    if dataset_name == "mnist":

        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

        # Convert 1-channel to 3-channel if needed
        if to_rgb:
            transform_list.insert(1, transforms.Grayscale(num_output_channels=3))

        transform_list.append(
            transforms.Normalize(
                mean=[0.5] * (3 if to_rgb else 1),
                std=[0.5] * (3 if to_rgb else 1),
            )
        )

        transform = transforms.Compose(transform_list)

        dataset_train = datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform
        )
        dataset_test = datasets.MNIST(
            root=data_root, train=False, download=True, transform=transform
        )

  

    elif dataset_name == "cifar10":

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
        ])

        dataset_train = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform
        )
        dataset_test = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform
        )


    elif dataset_name == "celeba":

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
        ])

        dataset_train  = datasets.ImageFolder(root="data/celeba",transform=transform)


        dataset_test = dataset_train

    elif dataset_name == "oneceleba":

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
        ])

        dataset_train  = datasets.ImageFolder(root="data/oneceleba",transform=transform)


        dataset_test = dataset_train

    elif dataset_name == "pokemon":

        transform = transforms.Compose([
            # transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
        ])

        dataset_train  = datasets.ImageFolder(root="data/pokemon",transform=transform)


        dataset_test = dataset_train

    

    else:
        raise ValueError("dataset_name must be one of: mnist, cifar10, celeba")


    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, test_loader

# def train_epoch(model,data_loader,optimizer,constants,device='cpu'):
#     model.train()
#     loss = nn.MSELoss()
#     device = torch.device(device if device else "cpu")
#     total_loss = 0
#     start_time = time.time()
#     pbar = tqdm(data_loader, desc="Training", leave=False)
#     for i, (img,label) in enumerate(data_loader):
#         img = img.to(device)
#         t = torch.randint(0,1000,(img.shape[0],)).to(device)
#         xt,noise  = q_sample(img,t,constants,device=device)

#         predicted_noise = model(xt,t)

#         l = loss(noise,predicted_noise)

#         optimizer.zero_grad()
#         l.backward()
#         optimizer.step()
#         total_loss += l.item()
   

#         if i % 10 == 0 and i > 0:

#             print(f"  Batch {i}/{len(data_loader)} | Current Loss: {l.item():.4f} | time taken: {time.time() - start_time:.2f}s")
#             start_time = time.time()
#         pbar.set_postfix(MSE=f"{l.item():.4f}")
#     return total_loss / len(data_loader)


def train_epoch(model, data_loader, optimizer, constants, device='cpu',ema = None):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0
    start_time = time.time()
    scaler = GradScaler(device=device)
    pbar = tqdm(data_loader, desc="Training", leave=False)
    

    for i, (img, label) in enumerate(pbar):
        img = img.to(device)
        
        
        t = torch.randint(0, 1000, (img.shape[0],)).to(device)
        
       
        xt, noise = q_sample(img, t, constants, device=device)
        with autocast(device_type=device):
            predicted_noise = model(xt, t)
            l = loss_fn(noise, predicted_noise)
  
        optimizer.zero_grad()
        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()
        # l.backward()
        # optimizer.step()
        if ema is not None:
            ema.update(model)
        
        total_loss += l.item()

        if i % 500 == 0 and i > 0:
            print(f"  Batch {i}/{len(data_loader)} | Current Loss: {l.item():.4f} | time taken: {time.time() - start_time:.2f}s")
            start_time = time.time()
        
        
        pbar.set_postfix(MSE=f"{l.item():.4f}")

    return total_loss / len(data_loader)

# def sample_img(model, constants, n_samples=1, device=None, img_Size=64):
#     device = torch.device(device if device else "cpu")
    
#     # 1. Start with pure noise
#     x = torch.randn((n_samples, 3, img_Size, img_Size)).to(device)
#     l = [x.cpu().numpy()] # Save initial noise frame
    
#     model.eval()
#     with torch.no_grad():
#         for t in reversed(range(1000)):
#             t_batch = torch.full((n_samples,), t, device=device).long()
            
#             
#             predicted_noise = model(x, t_batch).to(device)
#             predicted_noise = torch.clamp(predicted_noise, -1.0, 1.0) 
            
#            
#             alpha_t = constants["alpha"][t].to(device).view(-1, 1, 1, 1)
#             sqrt_alpha_t = constants["sqrt_alpha"][t].to(device).view(-1, 1, 1, 1)
#             sqrt_one_minus_alpha_bar_t = constants["sqrt_one_minus_alpha_bar"][t].to(device).view(-1, 1, 1, 1)
#             beta_t = constants["beta"][t].to(device).view(-1, 1, 1, 1)
            
#           
#             mean = 1 / sqrt_alpha_t * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_bar_t) * predicted_noise)

#             
#             if t > 0:
#                 z = torch.randn_like(x).to(device)
#                 sigma_t = torch.sqrt(beta_t) # Sigma is just sqrt(beta)
                
#                
#                 x = mean + sigma_t * z
#             else:
#                 x = mean
           
#             x = torch.clamp(x, -1.0, 1.0)
#             
#             if t % 20 == 0:
#                 # x_frame = torch.clamp(x, -1.0, 1.0)
#                 x_frame = (x + 1) / 2
#                 l.append(x_frame.cpu().numpy())

#    
#     x = (x + 1) / 2
#     return x, l



def sample_img(model, constants, n_samples=1, device=None, img_Size=64, seed=None):
    device = torch.device(device if device else "cpu")
    
    if seed is not None:
        torch.manual_seed(seed)

    #Start with pure noise
    x = torch.randn((n_samples, 3, img_Size, img_Size)).to(device)
    l = [x.cpu().numpy()] 
    
    model.eval()
    with torch.no_grad():
        for t in reversed(range(1000)):
            t_batch = torch.full((n_samples,), t, device=device).long()
            predicted_noise = model(x, t_batch)
            
            # get constants
            alpha_t = constants["alpha"][t].to(device).view(-1, 1, 1, 1)
            sqrt_alpha_t = constants["sqrt_alpha"][t].to(device).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = constants["sqrt_one_minus_alpha_bar"][t].to(device).view(-1, 1, 1, 1)
            posterior_var_t = constants["posterior_variance"][t].view(-1,1,1,1)
            sqrt_alpha_bar_t = constants["sqrt_alpha_bar"][t].to(device).view(-1, 1, 1, 1)

            x0_pred = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            corrected_noise = (x - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t
            
            # This is the standard formula: mean = 1/sqrt(alpha) * (x - coeff * noise)
            coeff = (1 - alpha_t) / sqrt_one_minus_alpha_bar_t
            mean = (1 / sqrt_alpha_t) * (x - coeff * corrected_noise)
            if t > 0:
                z = torch.randn_like(x).to(device)
                sigma_t = torch.sqrt(posterior_var_t) 
                x = mean + sigma_t * z
            else:
                x = mean

           

            # Save frame every 20 steps for gifs
            if t % 20 == 0:
  
                x_vis = (x + 1) / 2          
                l.append(x_vis.cpu().numpy())

    
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    return x, l

