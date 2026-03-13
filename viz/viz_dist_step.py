import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def viz_distribution(path="oMerge_logs/"):
    steps = list(range(32, 1024, 32))
    
    for step in steps:
        
        filepath = path + f"compress_distribution_{step}.pt"
        torch_data = torch.load(filepath)
        torch_data = torch_data[..., :1197] # here _ is the actual number of total tokens

        figpath = "./figs/comress_distribution/per_step/"
        os.makedirs(figpath, exist_ok=True)
        
        draw_data = torch_data[0, 0].cpu().numpy()
        plt.figure(figsize=(25, 10))
        plt.imshow(draw_data, cmap='viridis')
        plt.xlabel("sequence pos")
        plt.ylabel("cache pos")
        plt.tight_layout()
        plt.savefig(figpath + f"step_{step}_layer_{0}_head_{0}_com_dist.png")

if __name__ == "__main__":
    viz_distribution()