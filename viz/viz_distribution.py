import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def viz_distribution(path="oMerge_filter_logs/compress_distribution.pt"):
    torch_data = torch.load(path)
    
    torch_data = torch_data[..., :1197] # here _ is the actual number of total tokens

    figpath = "./figs/comress_distribution_filter/"
    os.makedirs(figpath, exist_ok=True)
    for layer_id in range(torch_data.shape[0]):
        for head_id in range(torch_data.shape[1]):
            draw_data = torch_data[layer_id, head_id].cpu().numpy()
            plt.figure(figsize=(25, 10))
            plt.imshow(draw_data, cmap='viridis')
            plt.xlabel("sequence pos")
            plt.ylabel("cache pos")
            plt.tight_layout()
            plt.savefig(figpath + f"layer_{layer_id}_head_{head_id}_com_dist.png")

if __name__ == "__main__":
    viz_distribution()