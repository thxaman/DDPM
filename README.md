# 🧠 Denoising Diffusion Probabilistic Model (DDPM) from Scratch

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

An unconditional Diffusion Model built **entirely from absolute mathematical scratch** in PyTorch. No HuggingFace `diffusers`, no pre-trained weights, no black-box APIs—just pure math, convolutional networks, and raw noise.

<img width="49%" height="100%" alt="diffusion_process (2)" src="https://github.com/user-attachments/assets/96dbb8dd-92c2-40f9-8409-bb19e0b3b042" /><img width="49%" height="100%" alt="final_sample" src="https://github.com/user-attachments/assets/d975d54d-3a16-4906-b150-5de0d001a0d7" /> 



## 📖 Overview
This project is a deep dive into the underlying architecture of modern Generative AI. It implements the seminal [DDPM paper](https://arxiv.org/abs/2006.11239) mechanics to learn the data distribution of the **CelebA (200,000 images)** dataset, sculpting highly detailed human faces out of pure Gaussian noise over 1,000 timesteps.

### ⚙️ Core Architecture & Features
* **Custom U-Net Backbone:** Engineered a deep 512-channel U-Net with dynamic GroupNorm logic.
* **Scaled Residual Connections:** Implemented `math.sqrt(2)` variance preservation to completely eliminate exploding gradients during deep forward passes.
* **Self-Attention Blocks:** Integrated Multi-Head Attention at the lower resolutions to capture global facial structures before upsampling.
* **EMA (Exponential Moving Average):** Built a custom shadow-weight class that decays at `0.995` per batch, effectively curing the "sandy/grainy noise" problem during inference.
* **$x_0$ Guardrail Sampling:** Upgraded the standard sampling algorithm to predict and clamp the clean $x_0$ image at every timestep, preventing values from spiraling to infinity without crushing contrast.

## 📉 The Training Journey
Building this was not without its hurdles. During early tests on a smaller dataset, the massive 512-channel U-Net actually outsmarted the MSE loss function. It realized that predicting pure white pixels yielded a mathematically safer score than attempting to draw complex shapes, resulting in complete mode collapse (The "White Blob of Death"). 

Scaling up to the massive, highly-varied **CelebA** dataset completely cured this, forcing the model to learn deep, rich latent representations of human features. 

### Watch the Network Learn (Pure Noise ➔ Face/Subject)
Old Celeb Face
<img width="1024" height="306" alt="evolution celeba old man (2)" src="https://github.com/user-attachments/assets/21865816-77ee-4980-846f-bebe723323be" />
Pokemon
<img width="1024" height="306" alt="image" src="https://github.com/user-attachments/assets/bd3f9f5f-512d-431f-97e8-e8d2d4dbe3bc" />

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/YOUR_USERNAME/DDPM-From-Scratch.git](https://github.com/thxaman/DDPM.git)
cd DDPM
pip install torch torchvision numpy matplotlib scipy tqdm
