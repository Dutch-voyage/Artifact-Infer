from zipfile import Path
import numpy as np
import matplotlib.pyplot as plt
import regex
import torch
import os
import torch
import torch.nn.functional as F
import glob
from pathlib import Path

import sqlite3
import json
import string
from rich import print as rprint

num_layers = 36
num_heads = 32
num_kv_heads = 8
window_size = 32

single_mode = True  # set to False to do diff mode

sampled_kv_heads = {0, 1, 2, 3}

if_abs = False

data_path="./test_test_logs"
fig_path="./test_figs"
process_path="./test_processed_logs"

if_save_ylim = False
if_use_ylim = True

path1 = f"./{data_path}/maxpool_num_topp_p90_512_iftemperatured_False_ifcompress_False.pt"
path2 = f"./{data_path}/maxpool_merge_lse_512_True.pt"

# path1 = "./test_logs/maxpool_lse_512_True.pt"
# path1 = "./test_logs/raw_num_topp_p60_512_False.pt"
postfix = "v1"

plt.rcParams.update({"font.size": 14})

def pooling_func_v2(x):
    """
    Calculates max and min pooling with replicate padding.
    Args:
        x: Input tensor of shape (Batch, Length)
    Returns:
        max_val, min_val: Tensors of shape (Batch, Output_Length)
    """
    x_lower = x.amin(-1).view(
        -1, num_layers * num_kv_heads
    )  # (num_steps, num_layers, num_kv_heads)
    x_lower = x_lower.transpose(0, 1).unsqueeze(
        1
    )  # (num_layers * num_kv_heads, num_steps)

    x_higher = x.amax(-1).view(
        -1, num_layers * num_kv_heads
    )  # (num_steps, num_layers, num_kv_heads)
    x_higher = x_higher.transpose(0, 1).unsqueeze(1)

    kernel_size = 8
    stride = 4

    pad_amount = kernel_size // 2
    x_lower_padded = F.pad(x_lower, (pad_amount, pad_amount), mode="replicate")
    x_higher_padded = F.pad(x_higher, (pad_amount, pad_amount), mode="replicate")

    min_val = F.avg_pool1d(x_lower_padded, kernel_size=kernel_size, stride=stride)

    max_val = F.avg_pool1d(x_higher_padded, kernel_size=kernel_size, stride=stride)

    # 5. Restore dimensions
    # Squeeze to remove the channel dimension: (B, 1, L_out) -> (B, L_out)
    return min_val.squeeze(1), max_val.squeeze(1)

def pooling_func_v1(x):
    """
    Calculates max and min pooling with replicate padding.
    Args:
        x: Input tensor of shape (Batch, Length)
    Returns:
        max_val, min_val: Tensors of shape (Batch, Output_Length)
    """
    x = x.view(-1, num_layers * num_kv_heads).to(
        torch.float32
    )  # (num_steps, num_layers, num_kv_heads)
    x = x.transpose(0, 1)  # (num_layers * num_kv_heads, num_steps)

    # 1. Setup parameters
    # x shape is (Batch, Length), but max_pool1d expects (Batch, Channel, Length)
    # We unsqueeze to add the channel dimension.
    x_input = x.unsqueeze(1)

    kernel_size = 8
    stride = 4

    # 2. Apply Padding
    # "Front and tail padded with nearest value" -> mode='replicate'
    # We pad (kernel_size // 2) on both sides to cover boundaries roughly evenly.
    # For kernel 32, this pads 16 elements on left and 16 on right.
    pad_amount = kernel_size // 2
    x_padded = F.pad(x_input, (pad_amount, pad_amount), mode="replicate")

    # 3. Calculate Max Pooling
    # We use the padded input directly
    max_val = F.max_pool1d(x_padded, kernel_size=kernel_size, stride=stride)

    # 4. Calculate Min Pooling
    # Method: -max_pool1d(-x)
    min_val = -F.max_pool1d(-x_padded, kernel_size=kernel_size, stride=stride)

    # 5. Restore dimensions
    # Squeeze to remove the channel dimension: (B, 1, L_out) -> (B, L_out)
    return min_val.squeeze(1), max_val.squeeze(1)


def preprocess(raw_path, savepath, metric_name, postfix):
    if "lse" in metric_name:
        preprocess_lse(raw_path, savepath, postfix)
    elif metric_name == "num_topp":
        preprocess_num_top_p(raw_path, savepath, postfix)

def preprocess_lse(raw_path, savepath, postfix):
    lse_data = torch.load(raw_path)
    lse_data = (
        torch.stack(lse_data, dim=0)
        .reshape(-1, num_layers, window_size, num_heads)
        .mean(-2)
    )  # (num_steps * num_layers, num_heads)
    lse_data = lse_data.view(-1, num_layers, num_kv_heads, num_heads // num_kv_heads)  #

    # lower, upper = pooling_func(lse_data)
    if postfix == "v1":
        lower, upper = pooling_func_v1(lse_data.mean(-1))
    elif postfix == "v2":
        lower, upper = pooling_func_v2(lse_data)

    lower = lower.transpose(0, 1).view(
        -1, num_layers, num_kv_heads
    )  # (num_steps, num_layers, num_kv_heads)
    upper = upper.transpose(0, 1).view(
        -1, num_layers, num_kv_heads
    )  # (num_steps, num_layers, num_kv_heads)

    # lower = lse_data.amin(-1)  # (num_steps, num_layers, num_kv_heads)
    # upper = lse_data.amax(-1)  # (num_steps, num_layers, num_kv_heads)

    np.save(savepath, {"lower": lower.cpu().numpy(), "upper": upper.cpu().numpy()})


def preprocess_num_top_p(raw_path, savepath, postfix):
    data = torch.load(raw_path)
    num_topp = torch.stack(data, dim=0).to(torch.float32)
    log_num_topp = torch.log(num_topp)
    log_num_topp = log_num_topp.reshape(-1, num_layers, num_kv_heads)

    assert postfix == "v1", "num_topp preprocessing only supports v1 pooling"
    lower, upper = pooling_func_v1(log_num_topp)
    lower = lower.transpose(0, 1).view(
        -1, num_layers, num_kv_heads
    )  # (num_steps, num_layers, num_kv_heads)
    upper = upper.transpose(0, 1).view(
        -1, num_layers, num_kv_heads
    )  # (num_steps, num_layers, num_kv_heads)

    # lower = lse_data.amin(-1)  # (num_steps, num_layers, num_kv_heads)
    # upper = lse_data.amax(-1)  # (num_steps, num_layers, num_kv_heads)

    np.save(savepath, {"lower": lower.cpu().numpy(), "upper": upper.cpu().numpy()})


def color_interpolate(start, mid, end, steps):
    from matplotlib.colors import to_rgb

    start_rgb = np.array(to_rgb(start))
    mid_rgb = np.array(to_rgb(mid))
    end_rgb = np.array(to_rgb(end))

    first_half = [
        start_rgb + (mid_rgb - start_rgb) * (i / (steps // 2))
        for i in range(steps // 2)
    ]
    second_half = [
        mid_rgb + (end_rgb - mid_rgb) * (i / (steps - steps // 2))
        for i in range(steps - steps // 2)
    ]
    color_list = first_half + second_half
    color_list = [tuple(color) for color in color_list]
    return color_list


def nrange(start, end, num):
    if num == 1:
        yield start
        return
    step = (end - start) / (num - 1)
    for i in range(num):
        yield start + i * step


def viz_legend(cu_fig_path=fig_path):
    """Generate a standalone legend figure showing the color scheme and line styles."""
    if cu_fig_path and not os.path.exists(cu_fig_path):
        os.makedirs(cu_fig_path)

    color_start = "#ff3b3b"
    color_mid = "#006994"
    color_end = "#8b0000"
    
    num_sample_kv_heads = len(sampled_kv_heads) if sampled_kv_heads is not None else num_kv_heads
    
    color_map = color_interpolate(color_start, color_mid, color_end, num_sample_kv_heads)

    color_map_alpha = [
        color_tuple + (0.3,) for color_tuple in color_map
    ]  # add alpha channel

    fig, ax = plt.subplots(figsize=(36, 1.2))

    # Create color gradient bar for 8 KV heads
    gradient = np.linspace(0, 1, num_sample_kv_heads).reshape(1, -1)
    ax.imshow(
        gradient,
        aspect="auto",
        cmap=plt.cm.colors.ListedColormap(color_map_alpha),
        extent=[0, num_sample_kv_heads, 0, 1],
    )

    # Add head labels and legend indicators for each color block
    for idx, head_id in enumerate(sampled_kv_heads if sampled_kv_heads is not None else range(num_kv_heads)):
        # Head label at the top
        ax.text(
            idx + 0.5,
            0.50,
            f"KV Head {head_id}",
            ha="center",
            va="center",
            fontsize=48,
            # fontweight="bold",
            color="black",
        )

        # # Draw solid line for Upper bound in the center of the block
        # ax.plot(
        #     [idx + 0.2, idx + 0.8],
        #     [0.35, 0.35],
        #     color=color_map[idx],
        #     linestyle="-",
        #     linewidth=4,
        # )

        # # Draw dashed line for Lower bound below the solid line
        # ax.plot(
        #     [idx + 0.2, idx + 0.8],
        #     [0.1, 0.1],
        #     color=color_map[idx],
        #     linestyle="--",
        #     linewidth=4,
        # )
        # # Add labels for the line styles

        # ax.text(
        #     idx + 0.5,
        #     0.5,
        #     "Upper",
        #     ha="center",
        #     va="center",
        #     fontsize=16,
        #     fontweight="bold",
        #     color="black",
        # )
        # ax.text(
        #     idx + 0.5,
        #     0.2,
        #     "Lower",
        #     ha="center",
        #     va="center",
        #     fontsize=16,
        #     fontweight="bold",
        #     color="black",
        # )

    ax.set_xlim(0, num_sample_kv_heads)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(f"{cu_fig_path}/lse_legend.png", dpi=300, bbox_inches="tight")
    plt.close()

def viz_single(save_path, cu_fig_path):
    if cu_fig_path and not os.path.exists(cu_fig_path):
        os.makedirs(cu_fig_path)
    lse_data = np.load(save_path, allow_pickle=True).item()
    lower = torch.tensor(lse_data["lower"])
    upper = torch.tensor(lse_data["upper"])

    viz_lower_upper(lower, upper, cu_fig_path)

def viz_diff(save_path1, save_path2, cu_fig_path):
    if cu_fig_path and not os.path.exists(cu_fig_path):
        os.makedirs(cu_fig_path)

    lse_data_1 = np.load(save_path1, allow_pickle=True).item()
    lower_1 = torch.tensor(lse_data_1["lower"])
    upper_1 = torch.tensor(lse_data_1["upper"])

    lse_data_2 = np.load(save_path2, allow_pickle=True).item()
    lower_2 = torch.tensor(lse_data_2["lower"])
    upper_2 = torch.tensor(lse_data_2["upper"])

    if if_abs:
        lower_diff = torch.abs(lower_1 - lower_2)
        upper_diff = torch.abs(upper_1 - upper_2)
    else:
        lower_diff = lower_1 - lower_2
        upper_diff = upper_1 - upper_2

    if if_save_ylim:
        export_y_lim(lower_diff, upper_diff, output_path="./tmp/y_lim_diff.npy")
    if if_use_ylim:
        y_lim_path = "./tmp/y_lim_diff.npy"
    else:
        y_lim_path = None
    
    viz_lower_upper(lower_diff, upper_diff, cu_fig_path, y_lim_path=y_lim_path)

def export_y_lim(lower, upper, output_path="y_lim.npy"):
    """
    Export y-axis limits for all layers based on lower and upper bounds.
    For each layer, computes the min/max across all heads and all time steps.
    Saves per-layer limits.

    Args:
        lower: Tensor of shape (num_steps, num_layers, num_kv_heads)
        upper: Tensor of shape (num_steps, num_layers, num_kv_heads)
        output_path: Path to save the y_lim dictionary
    """
    y_lim_dict = {}

    for layer_id in range(num_layers):
        # Collect all y-values being plotted for this layer (all heads, all steps)
        layer_lower = lower[:, layer_id, :]  # (num_steps, num_kv_heads)
        layer_upper = upper[:, layer_id, :]  # (num_steps, num_kv_heads)

        # Find actual min and max of all data being plotted
        y_min = min(layer_lower.min().item(), layer_upper.min().item())
        y_max = max(layer_lower.max().item(), layer_upper.max().item())

        # Add some padding (5% on each side)
        padding = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        y_min_padded = y_min - padding
        y_max_padded = y_max + padding

        y_lim_dict[layer_id] = (y_min_padded, y_max_padded)

    np.save(output_path, y_lim_dict)
    print(f"Y-limits exported to {output_path}")
    return y_lim_dict

def viz_lower_upper(lower, upper, cu_fig_path, ylabel=None, y_lim_path=None):
    if cu_fig_path and not os.path.exists(cu_fig_path):
        os.makedirs(cu_fig_path)

    # Load y_lim if path is provided
    y_lim_dict = None
    if y_lim_path is not None:
        try:
            y_lim_dict = np.load(y_lim_path, allow_pickle=True).item()
            print(f"Loaded y-limits from {y_lim_path}")
        except FileNotFoundError:
            print(f"Warning: y_lim_path {y_lim_path} not found, using automatic limits")

    color_start = "#ff3b3b"
    color_mid = "#006994"
    color_end = "#8b0000"
    color_map = color_interpolate(color_start, color_mid, color_end, num_kv_heads) if sampled_kv_heads is None else [
        color for idx, color in enumerate(color_interpolate(color_start, color_mid, color_end, num_kv_heads)) if idx in sampled_kv_heads
    ]

    for layer_id in range(num_layers):
        plt.figure(figsize=(8, 5))
        for head_id in range(num_kv_heads):
            if sampled_kv_heads is not None and head_id not in sampled_kv_heads:
                continue
            # make color darker for upper bound and lighter for lower bound
            plt.plot(
                list(nrange(512, 8192, lower.shape[0])),
                lower[:, layer_id, head_id].numpy(),
                color=color_map[head_id],
                linestyle="--",
                linewidth=1.5,
                label="Lower bound" if head_id == 0 else "",
            )
            plt.plot(
                list(nrange(512, 8192, lower.shape[0])),
                upper[:, layer_id, head_id].numpy(),
                color=color_map[head_id],
                linestyle="-",
                linewidth=1.5,
                label="Upper bound" if head_id == 0 else "",
            )
            plt.fill_between(
                list(nrange(512, 8192, lower.shape[0])),
                lower[:, layer_id, head_id].numpy(),
                upper[:, layer_id, head_id].numpy(),
                alpha=0.3,
                color=color_map[head_id],
            )
        plt.xlabel("Steps")
        if ylabel:
            plt.ylabel(ylabel)

        # Set y-limits if provided
        if y_lim_dict is not None and layer_id in y_lim_dict:
            y_min, y_max = y_lim_dict[layer_id]
            plt.ylim(y_min, y_max)

        # plt.legend()
        plt.savefig(
            f"{cu_fig_path}/layer_{layer_id}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

def filename_parse(filename):
    methods = ["maxpool_merge", "sim_merge", "raw", "maxpool", "sim"]
    metric_names = [
        "lse_topp",
        "num_topp",
        "lse",
        "selected_topp_indices",
        "temperatures_topp",
    ]
    ps = ["p60", "p70", "p80", "p90", "p95"]
    budgets = ["512"]
    for matched_method in methods:
        if matched_method in filename:
            method_type = matched_method
            break
    
    for matched_metric in metric_names:
        if "_" + matched_metric in filename:
            metric_name = matched_metric
            break
    
    p = None
    if "topp" in metric_name:
        for matched_p in ps:
            if matched_p in filename:
                p = matched_p
                break
    
    budget = 512
    for matched_budget in budgets:
        if "_" + matched_budget in filename:
            budget = matched_budget
            break
    
    if "iftemperatured" in filename:
        if_temperatured = regex.search(r"iftemperatured_(True|False)", filename).group(1)
    else:
        if_temperatured = "True"
    
    if "ifcompress" in filename:
        if_compress = regex.search(r"ifcompress_(True|False)", filename).group(1)
    else:
        if_compress = "True"

    return method_type, metric_name, p, budget, if_temperatured, if_compress

def setup_log_db():
    conn = sqlite3.connect("log_data.db")
    cursor = conn.cursor()
    files = glob.glob(f"./{data_path}/*.pt")

    # If you only want the filenames, not the full path:
    filenames = [os.path.basename(f) for f in files]

    
    "if table already exists, drop it"
    cursor.execute("DROP TABLE IF EXISTS file_index")
    cursor.execute(
        """
        CREATE TABLE file_index (
            raw_path TEXT,
            meta TEXT,
            save_path TEXT,
            fig_path TEXT
        )
    """
    )

    data = []

    for filename in filenames:
        method, metric_name, p, budget, if_temperatured, if_compress = filename_parse(filename)
        print(filename, method, metric_name, p, budget, if_temperatured, if_compress)
        metadata = {}
        metadata["method"] = method
        metadata["metric_name"] = metric_name
        if p:
            metadata["p"] = p
        metadata["budget"] = budget
        metadata["if_compress"] = if_compress
        metadata["if_temperatured"] = if_temperatured

        data.append((filename, json.dumps(metadata), None, None))

    cursor.executemany("INSERT INTO file_index (raw_path, meta, save_path, fig_path) VALUES (?, ?, ?, ?)", data)
    conn.commit()
    conn.close()


def raw_path_to_save_path(raw_path, metadata, postfix):
    conn = sqlite3.connect("log_data.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT meta FROM file_index WHERE raw_path = ?", (os.path.basename(raw_path),)
    )
    result = cursor.fetchone()

    meta = json.loads(result[0])

    save_path = "_".join(meta.values()) + f"_{postfix}.npy"

    # we will compute log num_topp instead of num_topp for visulizing metrics
    save_path = save_path.replace("num_topp", "log_num_topp")

    save_path = os.path.join(f"{process_path}", save_path)
    print(save_path)
    
    # add savepath as new column
    cursor.execute(
        "UPDATE file_index SET save_path = ? WHERE raw_path = ?",
        (save_path, raw_path),
    )
    conn.commit()
    conn.close()
    return save_path

if __name__ == "__main__":
    data_path="./test_test_logs"
    if data_path and not os.path.exists(data_path):
        assert False, f"data_path {data_path} does not exist"
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(process_path, exist_ok=True)    
    setup_log_db()
    
    raw_path = path1
    conn = sqlite3.connect("log_data.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT meta FROM file_index WHERE raw_path = ?", (os.path.basename(raw_path),)
    )
    result = cursor.fetchone()
    metadata = json.loads(result[0])

    rprint(metadata)
    
    cursor.execute("SELECT save_path FROM file_index WHERE raw_path = ?", (raw_path,))
    save_path = cursor.fetchone()
    if save_path is None:
        save_path = raw_path_to_save_path(raw_path, metadata, postfix)
        preprocess(raw_path, save_path, metadata["metric_name"], postfix)
    else:
        save_path = save_path[0]
        
    if single_mode:
        cu_fig_path = save_path.replace(".npy", "")
        cu_fig_path = cu_fig_path.replace(f"{process_path}/", fig_path + "/")
        print(cu_fig_path)
        cursor.execute(
            "UPDATE file_index SET fig_path = ? WHERE raw_path = ?",
            (cu_fig_path, raw_path),
        )
        viz_single(save_path, cu_fig_path)
    else:
        raw_path_2 = path2
        cursor.execute(
            "SELECT meta FROM file_index WHERE raw_path = ?", (os.path.basename(raw_path_2),)
        )
        result = cursor.fetchone()
        metadata_2 = json.loads(result[0])

        cursor.execute("SELECT save_path FROM file_index WHERE raw_path = ?", (raw_path_2,))
        save_path_2 = cursor.fetchone()

        if save_path_2 is None:
            save_path_2 = raw_path_to_save_path(raw_path_2, metadata_2, "v2")
            preprocess(raw_path_2, save_path_2, metadata_2["metric_name"], "v2")
        else:
            save_path_2 = save_path_2[0]
        
        if if_abs:
            metadata["method"] = "absdiff_" + metadata["method"] + "_vs_" + metadata_2["method"]
        else:
            metadata["method"] = "diff_" + metadata["method"] + "_vs_" + metadata_2["method"]
        
        cu_fig_path = "_".join(metadata.values())
        cu_fig_path = os.path.join(fig_path, cu_fig_path)
        print(cu_fig_path)
        
        viz_diff(save_path, save_path_2, cu_fig_path)
    conn.commit()
    conn.close()
    

