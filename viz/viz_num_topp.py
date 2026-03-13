import numpy as np
import matplotlib.pyplot as plt
import torch
import os
num_layers = 36
num_heads = 8
method_type = "sim"
p = 90
p1 = 95 
p2 = 60

def viz_num_topp_diff(path1=f"./viz_logs/{method_type}_num_topp_p{p1}.pt", path2=f"./viz_logs/{method_type}_num_topp_p{p2}.pt"):
    data1 = torch.load(path1)
    num_topp_1 = torch.stack(data1, dim=0).reshape(-1, num_layers, num_heads)
    data2 = torch.load(path2)
    num_topp_2 = torch.stack(data2, dim=0).reshape(-1, num_layers, num_heads)
    
    num_topp_diff = num_topp_1 - num_topp_2
    
    figpath = f"./figs/{method_type}_num_topp_diff_p{p1}_p{p2}"
    os.makedirs(figpath, exist_ok=True)
    steps = list(range(num_topp_diff.shape[0]))
    steps = [step * 32 + 512 for step in steps]
    for i in range(num_layers):
        data = num_topp_diff[:, i, :].cpu().numpy()
        fig = plt.figure(figsize=(15, 5))
        for head_i in range(num_heads):
            plt.plot(steps, data[:, head_i], label=f"head_{head_i}")
        plt.title("p999 - p99")
        plt.legend()
        plt.close()
        fig.savefig(f"{figpath}/layer_{i}.png")
        

def viz_num_topp(path=f"./test_logs/{method_type}_num_topp_p{p}.pt"):
    data = torch.load(path)
    num_topp = torch.stack(data, dim=0)
    num_topp = num_topp.reshape(-1, num_layers, num_heads)
    
    figpath = f"./figs/{method_type}_num_topp_p{p}"
    os.makedirs(figpath, exist_ok=True)
    steps = list(range(num_topp.shape[0]))
    
    steps = [step * 32 + 512 for step in steps]
    for i in range(num_layers):
        data = num_topp[:, i, :].cpu().numpy()
        fig = plt.figure(figsize=(15, 5))
        for head_i in range(num_heads):
            plt.plot(steps, data[:, head_i], label=f"head_{head_i}")
        plt.title(f"p{p}")
        plt.legend()
        plt.close()
        fig.savefig(f"{figpath}/layer_{i}.png")
        
def viz_temperatures(path=f"./test_logs/{method_type}_temperatures_p{p}.pt"):
    data = torch.load(path)
    num_topp = torch.stack(data, dim=0)
    num_topp = num_topp.reshape(-1, num_layers, num_heads)
    
    figpath = f"./figs/{method_type}_temperatures_p{p}"
    os.makedirs(figpath, exist_ok=True)
    steps = list(range(num_topp.shape[0]))
    
    steps = [step * 32 + 512 for step in steps]
    for i in range(num_layers):
        data = num_topp[:, i, :].cpu().numpy()
        fig = plt.figure(figsize=(15, 5))
        for head_i in range(num_heads):
            plt.plot(steps, data[:, head_i], label=f"head_{head_i}")
        plt.title(f"p{p}")
        plt.legend()
        plt.close()
        fig.savefig(f"{figpath}/layer_{i}.png")

def viz_selected_indices(path=f"./viz_logs/{method_type}_selected_topp_indices_p{p}.pt", num_topp_path=f"./viz_logs/{method_type}_num_topp_p{p}.pt"):
    data = torch.load(path)
    data_num = torch.load(num_topp_path)
    data_num = torch.stack(data_num, dim=0)
    
    data_num = data_num.reshape(-1, num_layers, num_heads)
        
    figpath = f"./figs/{method_type}_selected_indices_p{p}"
    os.makedirs(figpath, exist_ok=True)
    steps = [step * 32 + 512 for step in list(range(len(data) // num_layers))]
    
    for step_i in range(len(data) // num_layers):
        if step_i % 8 != 0:
             continue
        for layer_id in range(num_layers):
            i = step_i * num_layers + layer_id
            selected_indices = data[i]
            
            num = data_num[step_i][layer_id]
            heatmap = torch.zeros_like(selected_indices)
            indices_list = []
            for head_i in range(selected_indices.shape[0]):
                indices = selected_indices[head_i][:num[head_i]].tolist()
                
                # make tuple of (head_i, index)
                for idx in indices:
                    assert idx < heatmap.shape[-1]
                tupled_indices = [[head_i, idx] for idx in indices]
                indices_list.extend(tupled_indices)
            
            heatmap[tuple(zip(*indices_list))] = 1.0
            fig = plt.figure(figsize=(15, 5))
            plt.imshow(heatmap.cpu().numpy(), aspect='auto', cmap='viridis')
            ticks_length = (selected_indices.shape[-1] - 512) // 32
            plt.yticks(ticks=list(range(8)))
            plt.xticks(ticks=list(range(0, steps[ticks_length + 1], 32)))
            figpath_layer = f"{figpath}/layer_{layer_id}"
            os.makedirs(figpath_layer, exist_ok=True)
            fig.savefig(f"{figpath_layer}/step_{steps[step_i]}.png")
            plt.close()
    
    
if __name__ == "__main__":
    # viz_selected_indices()
    viz_num_topp()
    # viz_num_topp_diff()