import torch

from src.artifacts.nanovllm_v6.cache_mngr.oMerging_v2 import OrthMerging
from src.artifacts.nanovllm_v6.cache_mngr.oMerging import OrthMerging as OrthMerging_v1
from src.artifacts.nanovllm_v6.cache_mngr.snapKV_topp import SnapKV 
from src.services.nanovllm_v6.engine.sequence import Sequence

device = torch.device("cuda")

def test_orth_merging_compressor():
    # baseline = OrthMerging_v1(budget=512, window_size=32)
    # compressor = OrthMerging(budget=512, window_size=32)
    compressor = SnapKV(budget=512, window_size=32)

    # query_states = torch.randn(1, 32, 32, 128).to(device)  # (bsz, num_q_heads, q_len, head_dim)
    # key_states = torch.randn(1, 8, 512 + 32, 128).to(device)    # (bsz, num_kv_heads, kv_len, head_dim)
    # value_states = torch.randn(1, 8, 512 + 32, 128).to(device)  # (bsz, num_kv_heads, kv_len, head_dim)

    data = torch.load("./debug/snapKV_topp.pt")
    query_states = data["query_states"]
    key_states = data["key_states"]
    value_states = data["value_states"]
    
    seq = Sequence()
    seq.num_tokens = 526  # current kv length
    new_k, new_v = compressor.update_kv(query_states, key_states, value_states)
    
    # base_k, base_v = baseline.update_kv(query_states, key_states, value_states, seq, 0)
    
    print(new_k.shape)
    print(new_v.shape)

def main():
    test_orth_merging_compressor()

if __name__ == "__main__":
    main()