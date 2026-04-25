import math
import torch 
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn



class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dims):
        super().__init__()
        self.embedding_dims = embedding_dims

    def forward(self, t):
        device = t.device
        half_dim = self.embedding_dims // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # print(f"Time embedding shape: {embeddings.shape}")
        # print(f"Time embedding values: {embeddings}")
        return embeddings
    

class ResBlock(nn.Module):
    def __init__(self,in_channels, out_channels, time_embedding_dims,groups = 32):
        super().__init__()
        actual_groups_1 = groups if in_channels % groups == 0 else (8 if in_channels % 8 == 0 else 1)
        self.group_norm1 = nn.GroupNorm(actual_groups_1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.SiLU()
        self.time_layer = nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dims, out_channels))

        actual_groups_2 = groups if out_channels % groups == 0 else (8 if out_channels % 8 == 0 else 1)
        self.group_norm2 = nn.GroupNorm(actual_groups_2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.SiLU()

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        h = self.group_norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        t_emb_proj = self.time_layer(t_emb)[:,:,None,None]
        h = h + t_emb_proj

        h = self.group_norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return (self.residual_conv(x) + h)/math.sqrt(2)
    

class AttentionBlock(nn.Module):
        def __init__(self,in_channels,groups=32):
            super().__init__()   
            actual_groups = groups if in_channels % groups == 0 else (8 if in_channels % 8 == 0 else 1)
            self.group_norm = nn.GroupNorm(actual_groups, in_channels)
            self.mha = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)

        def forward(self,x):
            residual = x
            b,c,h,w = x.shape
            x = self.group_norm(x)
            x = x.view(b,c,h*w).transpose(1,2) # (b, h*w, c)
            attn_output, _ = self.mha(x,x,x)
            attn_output = attn_output.transpose(1,2).view(b,c,h,w)
            return (residual + attn_output)/math.sqrt(2)
        
        
class DiffusionUNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 3
        in_dims = 64
        time_dims = 512
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(embedding_dims=128),
            nn.Linear(128, time_dims),
            nn.SiLU(),
            nn.Linear(time_dims, time_dims)
        )

        self.init = nn.Conv2d(in_channels, in_dims, kernel_size=3, padding=1)

        self.down1_1 = ResBlock(in_dims, in_dims, time_embedding_dims=time_dims)
        self.down1_2 = ResBlock(in_dims, in_dims, time_embedding_dims=time_dims)
        self.pool1 = nn.Conv2d(in_dims, in_dims, kernel_size=3, stride=2, padding=1)


        self.down2_1 = ResBlock(in_dims, in_dims*2, time_embedding_dims=time_dims)
        self.down2_2 = ResBlock(in_dims*2, in_dims*2, time_embedding_dims=time_dims)
        self.attndown2 = AttentionBlock(in_dims*2)
        self.pool2 = nn.Conv2d(in_dims*2, in_dims*2, kernel_size=3, stride=2, padding=1)
        
        self.down3_1 = ResBlock(in_dims*2, in_dims*4, time_embedding_dims=time_dims)
        self.down3_2 = ResBlock(in_dims*4, in_dims*4, time_embedding_dims=time_dims)
        self.attndown3 = AttentionBlock(in_dims*4)
        self.pool3 = nn.Conv2d(in_dims*4, in_dims*4, kernel_size=3, stride=2, padding=1)

        self.bot1 = ResBlock(in_dims*4, in_dims*8, time_embedding_dims=time_dims)
        self.attnbot1 = AttentionBlock(in_dims*8)
        self.bot2 = ResBlock(in_dims*8, in_dims*8, time_embedding_dims=time_dims)

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_dims*8, in_dims*4, kernel_size=3, padding=1))
        self.resup3_1 = ResBlock(2*in_dims*4,in_dims*4 , time_embedding_dims=time_dims)
        self.resup3_2 = ResBlock(2*in_dims*4, in_dims*4, time_embedding_dims=time_dims)
        self.attnup3 = AttentionBlock(in_dims*4)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_dims*4, in_dims*2, kernel_size=3, padding=1))
        self.resup2_1 = ResBlock(2*in_dims*2,in_dims*2 , time_embedding_dims=time_dims)
        self.resup2_2 = ResBlock(2*in_dims*2, in_dims*2, time_embedding_dims=time_dims)
        self.attnup2 = AttentionBlock(in_dims*2)

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_dims*2, in_dims, kernel_size=3, padding=1))
        self.resup1_1 = ResBlock(2*in_dims,in_dims , time_embedding_dims=time_dims)
        self.resup1_2 = ResBlock(2*in_dims, in_dims, time_embedding_dims=time_dims)
      
        self.final_norm = nn.GroupNorm(32,in_dims)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(in_dims, in_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init(x)
        skip = []

        x = self.down1_1(x, t_emb)
        skip.append(x)
        x = self.down1_2(x, t_emb)
        skip.append(x)
        x = self.pool1(x)

        x = self.down2_1(x, t_emb)
        skip.append(x)
        x = self.down2_2(x, t_emb)
        x = self.attndown2(x)
        skip.append(x)
        x = self.pool2(x)

        x = self.down3_1(x, t_emb)
        skip.append(x)
        x = self.down3_2(x, t_emb)
        x = self.attndown3(x)
        skip.append(x)
        x = self.pool3(x)

        x = self.bot1(x, t_emb)
        x = self.attnbot1(x)
        x = self.bot2(x, t_emb)

        x = self.up3(x)
        x = torch.cat([x, skip.pop()], dim=1)
        x = self.resup3_1(x, t_emb)
        x = torch.cat([x, skip.pop()], dim=1)
        x = self.resup3_2(x, t_emb)
        x = self.attnup3(x)

        x = self.up2(x)
        x = torch.cat([x, skip.pop()], dim=1)
        x = self.resup2_1(x, t_emb)
        x = torch.cat([x, skip.pop()], dim=1)
        x = self.resup2_2(x, t_emb)
        x = self.attnup2(x)

        x = self.up1(x)
        x = torch.cat([x, skip.pop()], dim=1)
        x = self.resup1_1(x, t_emb)
        x = torch.cat([x, skip.pop()], dim=1)
        x = self.resup1_2(x, t_emb)


        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_conv(x)


        return x