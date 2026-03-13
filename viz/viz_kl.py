from matplotlib import pyplot as plt
import numpy as np
import os

def viz_kl(fig_path):
    names = ["RKV", "oMerge_filter", "oMerge", "SnapKV"]
    paths = {"RKV": "rkv_logs", "oMerge_filter": "oMerge_filter_logs", "oMerge": "oMerge_logs", "SnapKV": "snapkv_logs"}
    for name in names:
        kl_data = np.load(f"{paths[name]}/kl.npy", allow_pickle=True).item()
        steps = kl_data["steps"]
        kls = kl_data["kls"]
        if_match = kl_data.get("if_match")
        print(f"{name} total samples: {len(if_match)}, match: {sum(if_match)}")
        kls = np.array(kls)
        
        plt.plot(steps, np.log10(np.where(kls > 0, kls, 1e-16)), label=name)
    
    plt.xlabel("Steps")
    plt.ylabel("log KL Divergence")
    plt.title("KL Divergence over Steps")
    plt.legend()
    if fig_path and not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(f"{fig_path}/kl_comparison.png")

if __name__ == "__main__":
    viz_kl("figs/kld")