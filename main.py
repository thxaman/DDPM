import os
import glob
import torch
from better_model import * 
from helper_funcitons import *

# just a function to get paths, save models and chkpoints 
def get_run_folder(dataset_name, resume=False):
    """Finds the latest run folder to resume, or creates a new one."""
    base = "models"
    file_path = os.path.join(base, dataset_name)
    os.makedirs(file_path, exist_ok=True)
    
    existing_runs = [d for d in os.listdir(file_path) if d.startswith("run_")]
    
    if resume and existing_runs:
        latest_run_id = max(int(i.split("_")[-1]) for i in existing_runs)
        run_name = os.path.join(file_path, f"run_{latest_run_id}")
        print(f"--> Resuming in existing folder: {run_name}")
        return run_name
    else:
        run_id = max([int(i.split("_")[-1]) for i in existing_runs]) + 1 if existing_runs else 0
        run_name = os.path.join(file_path, f"run_{run_id}")
        os.makedirs(run_name, exist_ok=True)
        os.makedirs(os.path.join(run_name, "checkpoints"), exist_ok=True)
        print(f"--> Created new run folder: {run_name}")
        return run_name



def main():
    
    dataset_name = "celeba"
    epochs = 50 
    batch_size = 32
    RESUME_TRAINING = False # False if you want to force a fresh restart
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = DiffusionUNetModel().to(device)
    ema = EMA(model=model, decay=0.995)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    constants = get_ddpm_constants(device=device)
    
    train_loader, _ = get_data(dataset_name=dataset_name, batch_size=batch_size, num_workers=8, to_rgb=True)
    
    model_path = get_run_folder(dataset_name, resume=RESUME_TRAINING)
    start_epoch = 0
    
    if RESUME_TRAINING:
        ckpt_dir = os.path.join(model_path, "checkpoints")
       
        ckpts = glob.glob(os.path.join(ckpt_dir, "unified_ckpt_epoch_*.pth"))
        
        if ckpts:
           
            latest_ckpt = sorted(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            print(f"--> Loading checkpoint: {latest_ckpt}")
            
            checkpoint = torch.load(latest_ckpt, map_location=device)
            
         
            model.load_state_dict(checkpoint['model_state_dict'])
            ema.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"--> Successfully resumed from Epoch {start_epoch}. Optimizer momentum preserved.")
        else:
            print("--> No checkpoints found. Starting from scratch.")

    # Training Loop
    for epoch in range(start_epoch, epochs):
        
        avg_loss = train_epoch(model, train_loader, optimizer, constants, device, ema=ema)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.5f}")

        # Save checkpoint every n epochs
        if (epoch + 1) % 5 == 0:
            
            unified_ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            ckpt_file = f"{model_path}/checkpoints/unified_ckpt_epoch_{epoch+1}.pth"
            torch.save(unified_ckpt, ckpt_file)
            
            ema_file = f"{model_path}/checkpoints/model_weights_({dataset_name})_ema.pth"
            torch.save(ema.ema_model.state_dict(), ema_file)
            print(f"--> Saved Checkpoint for Epoch {epoch+1}")

    torch.save(model.state_dict(), f"{model_path}/model_weights_({dataset_name})_FINAL.pth")
    torch.save(ema.ema_model.state_dict(), f"{model_path}/model_weights_({dataset_name})_ema_FINAL.pth")
    print(f"Training Complete! Final model saved to {model_path}")

if __name__ == "__main__":
    main()