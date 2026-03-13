import numpy
import matplotlib.pyplot as plt
import torch
import os

num_layers = 36
num_heads = 32
def viz_lse(path, fig_path):
    
    if fig_path and not os.path.exists(fig_path):
        os.makedirs(fig_path)
    lse_data = torch.load(path)
    lse_data = torch.stack(lse_data, dim=0).reshape(-1, num_layers, num_heads).cpu().numpy()  # (num_steps * num_layers, num_heads)

    for layer_id in range(num_layers):
        for head_id in range(num_heads):
            plt.figure(figsize=(20, 5))
            plt.plot(lse_data[:, layer_id, head_id], label=f"head {head_id}")
            plt.title(f"Layer {layer_id} Head {head_id} LSE over time")
            plt.xlabel("Steps")
            plt.ylabel("LSE")
            plt.legend()
            plt.savefig(f"{fig_path}/layer_{layer_id}_head_{head_id}_lse.png")
            plt.close()

if __name__ == "__main__":
    viz_lse("oMerge_v2_logs/lse_log.pt", "figs/oMerge_v2")